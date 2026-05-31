import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartAttention

from shared.kernels import (
    band_attention,
    build_key_mask,
    elu_linear_attention,
    should_use_triton,
)

# Idea 2: Lightning Hybrid (Intra-block Softmax + Inter-block Linear)
# We partition attention into a sharp local block (via masked softmax) 
# and an efficient linear attention for all long-range context.

class LightningHybridAttention(BartAttention):
    def __init__(self, base_attn: BartAttention, block_size: int = 64, use_triton: bool = True):
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
        self.block_size = block_size
        self.use_triton = use_triton

    def _use_triton_kernels(self, q: torch.Tensor) -> bool:
        return should_use_triton(self.use_triton, q, training=self.training)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _local_branch(self, Q, K, V, bsz, tgt_len, src_len, attention_mask, am_bool):
        """Sharp local (symmetric band) softmax attention."""
        if self._use_triton_kernels(Q):
            try:
                key_mask = build_key_mask(attention_mask, bsz, self.num_heads, src_len, Q.device)
                return band_attention(Q, K, V, self.block_size // 2, key_mask, scale=1.0)
            except Exception:
                pass

        BH = Q.size(0)
        scores = torch.bmm(Q, K.transpose(1, 2))
        idx_q = torch.arange(tgt_len, device=Q.device).unsqueeze(1)
        idx_k = torch.arange(src_len, device=K.device).unsqueeze(0)
        local_mask = (torch.abs(idx_q - idx_k) <= self.block_size // 2).unsqueeze(0).expand(BH, -1, -1)
        if am_bool is not None:
            mask_expanded = am_bool.expand(bsz, self.num_heads, tgt_len, src_len).reshape(BH, tgt_len, src_len)
            local_mask = local_mask & mask_expanded
        local_scores = scores.masked_fill(~local_mask, -1e9)
        local_attn = F.softmax(local_scores, dim=-1)
        local_attn = F.dropout(local_attn, p=self.dropout, training=self.training)
        return torch.bmm(local_attn, V)

    def _global_branch(self, Q, K, V, bsz, src_len, am_bool):
        """Linear attention over the full sequence (ELU feature map)."""
        BH = Q.size(0)
        if self._use_triton_kernels(Q):
            try:
                key_mask = None
                if am_bool is not None:
                    pad = am_bool.reshape(bsz, -1, src_len)[:, 0, :]
                    key_mask = pad.unsqueeze(1).expand(bsz, self.num_heads, src_len).reshape(BH, src_len)
                return elu_linear_attention(Q, K, V, key_mask)
            except Exception:
                pass

        Q_l = F.elu(Q) + 1.0
        K_l = F.elu(K) + 1.0
        if am_bool is not None:
            padding_mask = am_bool.reshape(bsz, -1, src_len)[:, 0, :]
            padding_mask = padding_mask.unsqueeze(1).expand(bsz, self.num_heads, src_len).reshape(BH, src_len)
            K_l = K_l * padding_mask.unsqueeze(-1)
        KV = torch.bmm(K_l.transpose(1, 2), V)
        Z = K_l.sum(dim=1, keepdim=True)
        Num = torch.bmm(Q_l, KV)
        Den = torch.bmm(Q_l, Z.transpose(1, 2))
        return Num / (Den + 1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * (self.head_dim ** -0.5)
        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        BH = bsz * self.num_heads
        Q = self._shape(query_states, tgt_len, bsz).reshape(BH, tgt_len, self.head_dim)
        K = key_states.reshape(BH, -1, self.head_dim)
        V = value_states.reshape(BH, -1, self.head_dim)
        src_len = K.size(1)

        am_bool = None
        if attention_mask is not None:
            am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
            if am_bool.dim() == 2:
                am_bool = am_bool[:, None, None, :]

        # 1. Local Block Attention (Intra-block) Softmax
        local_out = self._local_branch(Q, K, V, bsz, tgt_len, src_len, attention_mask, am_bool)

        # 2. Linear Attention (Inter-block long-range context)
        global_out = self._global_branch(Q, K, V, bsz, src_len, am_bool)

        # Combine the high-fidelity local output with the linear global summary
        out = local_out + 0.5 * global_out

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim) \
                        .transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, None)

def patch_bart(model: nn.Module, block_size: int = 64, use_triton: bool = True):
    def _recurse(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, BartAttention):
                if getattr(child, "is_decoder", False): continue
                setattr(module, name, LightningHybridAttention(child, block_size, use_triton=use_triton))
            else:
                _recurse(child)
    _recurse(model)

class AttnPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj  = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.proj(x))               
        s = self.score(h).squeeze(-1)              
        s = s.masked_fill(~mask.bool(), torch.finfo(s.dtype).min)
        a = torch.softmax(s, dim=-1)               
        return torch.bmm(a.unsqueeze(1), x).squeeze(1)

from transformers.modeling_outputs import SequenceClassifierOutput

class PatchedModel(nn.Module):
    def __init__(self, base_model, block_size=64, use_triton=True):
        super().__init__()
        self.model = base_model
        self.use_triton = use_triton
        patch_bart(self.model, block_size=block_size, use_triton=use_triton)
        hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model")
        self.attn_pool = AttnPool(hidden_size)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
        )
        last_hidden = outputs.last_hidden_state   
        if attention_mask is None:
            if input_ids is not None and self.model.config.pad_token_id is not None:
                attention_mask = (input_ids != self.model.config.pad_token_id).long()
            else:
                attention_mask = torch.ones(last_hidden.size()[:2], device=last_hidden.device, dtype=torch.long)
        
        pooled = self.attn_pool(last_hidden, attention_mask)
        logits = self.model.classification_head(pooled)

        loss = None
        if labels is not None:
            if labels.dtype != torch.long: labels = labels.long()
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return SequenceClassifierOutput(loss=loss, logits=logits)
