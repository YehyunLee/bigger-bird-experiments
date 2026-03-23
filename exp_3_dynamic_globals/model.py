import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartAttention

# Idea: Dynamic Content-Aware Globals
# Instead of fixed CLS tokens or random tokens, a lightweight gating network 
# decides which tokens are currently 'global' and should be broadcast 
# to all other tokens. The gate runs in linear time. We combine this with a standard sliding window.

class DynamicGlobalAttention(BartAttention):
    def __init__(self, base_attn: BartAttention, window_size: int = 64, num_globals: int = 16):
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
        
        self.window_size = window_size
        self.num_globals = num_globals
        
        # New: Gate to predict token "globalness" directly from inputs
        self.global_gate = nn.Linear(self.embed_dim, 1)

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

        if src_len <= (self.window_size * 2 + self.num_globals):
            # Dense fallback
            scores = torch.bmm(Q, K.transpose(1, 2))
            if attention_mask is not None:
                am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if am_bool.dim() == 2: am_bool = am_bool[:, None, None, :]
                mask_expanded = am_bool.expand(bsz, self.num_heads, tgt_len, src_len).reshape(BH, tgt_len, src_len)
                scores = scores.masked_fill(~mask_expanded, -1e9)
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            out = torch.bmm(attn_probs, V).reshape(BH, tgt_len, self.head_dim)
        else:
            # 1. Global Selection (O(N) operation)
            # Evaluate globalness from hidden_states
            global_scores = self.global_gate(hidden_states).squeeze(-1) # [B, T]
            
            if attention_mask is not None:
                global_mask = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if global_mask.dim() == 4: global_mask = global_mask.squeeze(1).squeeze(1) # [B, T]
                global_scores = global_scores.masked_fill(~global_mask, -1e9)
            
            _, global_idx = torch.topk(global_scores, k=self.num_globals, dim=-1) # [B, G]
            
            # Expand to all heads
            global_idx_exp = global_idx.unsqueeze(1).expand(bsz, tgt_len, self.num_globals)
            global_idx_exp = global_idx_exp.unsqueeze(1).expand(bsz, self.num_heads, tgt_len, self.num_globals).reshape(BH, tgt_len, self.num_globals)

            # 2. Local Window Selection (O(N) operation)
            t = torch.arange(tgt_len, device=Q.device)
            window_starts = torch.clamp(t - self.window_size // 2, min=0, max=src_len - self.window_size)
            offsets = torch.arange(self.window_size, device=Q.device)
            local_idx = window_starts.unsqueeze(1) + offsets.unsqueeze(0) # [Tq, W]
            local_idx_exp = local_idx.unsqueeze(0).expand(BH, -1, -1) # [BH, Tq, W]
            
            # Combine absolute indices
            abs_idx = torch.cat([global_idx_exp, local_idx_exp], dim=-1) # [BH, Tq, G + W]
            M = abs_idx.size(-1)

            # 3. Gather and Compute Attention
            idx_gather = abs_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            K_sel = torch.gather(K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_gather)
            V_sel = torch.gather(V.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_gather)

            scores_sel = torch.matmul(Q.unsqueeze(2), K_sel.transpose(-1, -2)).squeeze(2) # [BH, Tq, M]
            
            if attention_mask is not None:
                am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if am_bool.dim() == 2: am_bool = am_bool[:, None, None, :]
                am_small = am_bool.expand(bsz, 1, tgt_len, src_len)
                abs_idx_hb = abs_idx.view(self.num_heads, bsz, tgt_len, M)
                
                allowed_chunks = []
                for h in range(self.num_heads):
                    allowed_h = torch.gather(am_small, -1, abs_idx_hb[h].unsqueeze(1)).squeeze(1) # [B, Tq, M]
                    allowed_chunks.append(allowed_h)
                allowed = torch.cat(allowed_chunks, dim=0) # [BH, Tq, M]
                scores_sel = scores_sel.masked_fill(~allowed, -1e9)
            
            attn_probs = F.softmax(scores_sel, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            
            out = torch.bmm(
                attn_probs.reshape(BH * tgt_len, 1, M),
                V_sel.reshape(BH * tgt_len, M, self.head_dim)
            ).reshape(BH, tgt_len, self.head_dim)

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim) \
                        .transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if use_cache:
            return (attn_output, None, (key_states, value_states))
        return (attn_output, None)

def patch_bart(model: nn.Module, window_size: int = 64, num_globals: int = 16):
    def _recurse(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, BartAttention):
                if getattr(child, "is_decoder", False):
                    continue
                setattr(module, name, DynamicGlobalAttention(child, window_size, num_globals))
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
    def __init__(self, base_model, window_size=64, num_globals=16):
        super().__init__()
        self.model = base_model
        patch_bart(self.model, window_size=window_size, num_globals=num_globals)
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
            
        return SequenceClassifierOutput(loss=loss, logits=logits)
    
    @property
    def config(self):
        return self.model.config
