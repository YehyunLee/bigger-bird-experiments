import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartAttention
from transformers.modeling_outputs import SequenceClassifierOutput

# Idea C: Layer-Adaptive Sparsity
# Different layers have different roles:
#   - Early layers (1-3): syntactic, local patterns -> dense / very light sparse
#   - Middle layers (4-8): semantic composition -> moderate top-k
#   - Late layers (9-12): high-level reasoning -> heavy sparsity + globals
# Each layer gets its own (top_k) value based on its depth.


class LayerAdaptiveAttention(BartAttention):
    """DeepSeek-style top-k attention but with PER-LAYER k.
    Layer index is stored as an attribute, set by patch_bart based on enumeration order."""
    def __init__(self, base_attn: BartAttention, top_k: int = 64, low_rank_dim: int = 16, layer_idx: int = -1):
        super().__init__(
            embed_dim=base_attn.embed_dim,
            num_heads=base_attn.num_heads,
            dropout=base_attn.dropout.p if isinstance(base_attn.dropout, nn.Dropout) else float(base_attn.dropout),
            is_decoder=base_attn.is_decoder,
            bias=base_attn.k_proj.bias is not None,
        )
        self.q_proj.load_state_dict(base_attn.q_proj.state_dict())
        self.k_proj.load_state_dict(base_attn.k_proj.state_dict())
        self.v_proj.load_state_dict(base_attn.v_proj.state_dict())
        self.out_proj.load_state_dict(base_attn.out_proj.state_dict())
        self.top_k = top_k
        self.low_rank_dim = low_rank_dim
        self.layer_idx = layer_idx

    def _shape(self, tensor, seq_len, bsz):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                use_cache=False, **kwargs):
        for k in ("cache_position", "position_bias", "alibi_bias"):
            kwargs.pop(k, None)

        is_cross = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        Qs = self.q_proj(hidden_states) * (self.head_dim ** -0.5)
        if is_cross:
            Ks = self._shape(self.k_proj(key_value_states), -1, bsz)
            Vs = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            Ks = self._shape(self.k_proj(hidden_states), -1, bsz)
            Vs = self._shape(self.v_proj(hidden_states), -1, bsz)
        BH = bsz * self.num_heads
        Q = self._shape(Qs, tgt_len, bsz).reshape(BH, tgt_len, self.head_dim)
        K = Ks.reshape(BH, -1, self.head_dim)
        V = Vs.reshape(BH, -1, self.head_dim)
        src_len = K.size(1)

        if src_len <= self.top_k:
            scores = torch.bmm(Q, K.transpose(1, 2))
            if attention_mask is not None:
                am = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if am.dim() == 2: am = am[:, None, None, :]
                me = am.expand(bsz, self.num_heads, tgt_len, src_len).reshape(BH, tgt_len, src_len)
                scores = scores.masked_fill(~me, -1e9)
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.bmm(attn, V).reshape(BH, tgt_len, self.head_dim)
        else:
            d_low = min(self.low_rank_dim, self.head_dim)
            rough = torch.bmm(Q[:, :, :d_low], K[:, :, :d_low].transpose(1, 2)) / (d_low ** 0.5)
            if attention_mask is not None:
                am = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if am.dim() == 2: am = am[:, None, None, :]
                me = am.expand(bsz, self.num_heads, tgt_len, src_len).reshape(BH, tgt_len, src_len)
                rough = rough.masked_fill(~me, -1e9)
            _, idx = torch.topk(rough, k=self.top_k, dim=-1)
            idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            K_sel = torch.gather(K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_e)
            V_sel = torch.gather(V.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_e)
            s_sel = torch.matmul(Q.unsqueeze(2), K_sel.transpose(-1, -2)).squeeze(2)
            attn = F.softmax(s_sel, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.bmm(
                attn.reshape(BH * tgt_len, 1, self.top_k),
                V_sel.reshape(BH * tgt_len, self.top_k, self.head_dim)
            ).reshape(BH, tgt_len, self.head_dim)

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        return (self.out_proj(attn_output), None)


def _schedule(layer_idx: int, n_layers: int, k_early: int, k_mid: int, k_late: int):
    """Return top_k for a layer based on its depth."""
    third = n_layers / 3
    if layer_idx < third:
        return k_early
    elif layer_idx < 2 * third:
        return k_mid
    else:
        return k_late


def patch_bart(model: nn.Module, k_early=192, k_mid=64, k_late=32, low_rank_dim=16, n_layers_hint=12):
    """Walks BartEncoder layers and assigns per-layer top_k based on depth."""
    # Find encoder layers
    encoder = getattr(model, "model", model)
    if hasattr(encoder, "encoder"):
        encoder = encoder.encoder
    layers = getattr(encoder, "layers", None)
    if layers is None:
        # Fallback: just recurse and treat all attentions as middle
        n = n_layers_hint
        idx_holder = {"i": 0}
        def _rec(m):
            for nm, c in list(m.named_children()):
                if isinstance(c, BartAttention) and not getattr(c, "is_decoder", False):
                    k = _schedule(idx_holder["i"], n, k_early, k_mid, k_late)
                    setattr(m, nm, LayerAdaptiveAttention(c, top_k=k, low_rank_dim=low_rank_dim, layer_idx=idx_holder["i"]))
                    idx_holder["i"] += 1
                else:
                    _rec(c)
        _rec(model)
        return idx_holder["i"]

    n = len(layers)
    for i, layer in enumerate(layers):
        k = _schedule(i, n, k_early, k_mid, k_late)
        # Replace self_attn
        sa = layer.self_attn
        layer.self_attn = LayerAdaptiveAttention(sa, top_k=k, low_rank_dim=low_rank_dim, layer_idx=i)
    return n


class AttnPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, x, mask):
        h = torch.tanh(self.proj(x))
        s = self.score(h).squeeze(-1)
        s = s.masked_fill(~mask.bool(), torch.finfo(s.dtype).min)
        a = torch.softmax(s, dim=-1)
        return torch.bmm(a.unsqueeze(1), x).squeeze(1)


class PatchedModel(nn.Module):
    def __init__(self, base_model, k_early=192, k_mid=64, k_late=32, low_rank_dim=16):
        super().__init__()
        self.model = base_model
        patch_bart(self.model, k_early=k_early, k_mid=k_mid, k_late=k_late, low_rank_dim=low_rank_dim)
        hidden = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model")
        self.attn_pool = AttnPool(hidden)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(**kwargs)
    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()
    @property
    def supports_gradient_checkpointing(self):
        return getattr(self.model, "supports_gradient_checkpointing", True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.model.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last = out.last_hidden_state
        if attention_mask is None:
            if input_ids is not None and self.model.config.pad_token_id is not None:
                attention_mask = (input_ids != self.model.config.pad_token_id).long()
            else:
                attention_mask = torch.ones(last.size()[:2], device=last.device, dtype=torch.long)
        pooled = self.attn_pool(last, attention_mask)
        logits = self.model.classification_head(pooled)
        loss = None
        if labels is not None:
            if labels.dtype != torch.long: labels = labels.long()
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)

    @property
    def config(self):
        return self.model.config
