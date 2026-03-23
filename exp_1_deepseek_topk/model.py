import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartAttention

# We implement a Top-K sparse attention inspired by DeepSeek's Lightning Indexer.
# For efficiency in Python without custom kernels, we approximate the indexer by computing 
# a low-rank score matrix O(N^2 * D_{low}), fetching the top K, and doing actual attention over K.

class DeepSeekTopKAttention(BartAttention):
    def __init__(self, base_attn: BartAttention, top_k: int = 128, low_rank_dim: int = 16):
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

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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
        _ = kwargs.pop("cache_position", None)
        _ = kwargs.pop("position_bias", None)
        _ = kwargs.pop("alibi_bias", None)

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

        # Skip routing if sequence is short
        if src_len <= self.top_k:
            scores = torch.bmm(Q, K.transpose(1, 2))
            if attention_mask is not None:
                am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if am_bool.dim() == 2: am_bool = am_bool[:, None, None, :]
                mask_expanded = am_bool.expand(bsz, self.num_heads, tgt_len, src_len).reshape(BH, tgt_len, src_len)
                scores = scores.masked_fill(~mask_expanded, torch.finfo(scores.dtype).min)
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            out = torch.bmm(attn_probs, V).reshape(BH, tgt_len, self.head_dim)
        else:
            # --- The DeepSeek Lightning Indexer approximate approach ---
            d_low = min(self.low_rank_dim, self.head_dim)
            Q_low = Q[:, :, :d_low]
            K_low = K[:, :, :d_low]
            
            # Rough routing scores -> much faster matmul
            rough_scores = torch.bmm(Q_low, K_low.transpose(1, 2)) / (d_low ** 0.5)
            
            if attention_mask is not None:
                am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if am_bool.dim() == 2: am_bool = am_bool[:, None, None, :]
                mask_expanded = am_bool.expand(bsz, self.num_heads, tgt_len, src_len).reshape(BH, tgt_len, src_len)
                rough_scores = rough_scores.masked_fill(~mask_expanded, -1e9)
            
            # Extract top K
            _, topk_indices = torch.topk(rough_scores, k=self.top_k, dim=-1) # [BH, Tq, K]
            
            # Gather precise K and V
            idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            K_sel = torch.gather(K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_exp)
            V_sel = torch.gather(V.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_exp)

            # Precise score calc
            scores_sel = torch.matmul(Q.unsqueeze(2), K_sel.transpose(-1, -2)).squeeze(2) # [BH, Tq, K]
            
            attn_probs = F.softmax(scores_sel, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            
            # Multiply by V
            out = torch.bmm(
                attn_probs.reshape(BH * tgt_len, 1, self.top_k),
                V_sel.reshape(BH * tgt_len, self.top_k, self.head_dim)
            ).reshape(BH, tgt_len, self.head_dim)

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim) \
                        .transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if use_cache:
            return (attn_output, None, (key_states, value_states))
        return (attn_output, None)

def patch_bart(model: nn.Module, top_k: int = 128, low_rank_dim: int = 16):
    def _recurse(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, BartAttention):
                if getattr(child, "is_decoder", False):
                    continue
                setattr(module, name, DeepSeekTopKAttention(child, top_k, low_rank_dim))
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

class PatchedModel(nn.Module):
    def __init__(self, base_model, top_k=128):
        super().__init__()
        self.model = base_model
        patch_bart(self.model, top_k=top_k)
        hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model")
        self.attn_pool = AttnPool(hidden_size)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"): return self.model.gradient_checkpointing_enable(**kwargs)
    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"): return self.model.gradient_checkpointing_disable()
    @property
    def supports_gradient_checkpointing(self): return getattr(self.model, "supports_gradient_checkpointing", True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
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
            
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)
    
    @property
    def config(self):
        return self.model.config
