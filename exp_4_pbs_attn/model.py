import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartAttention

# Idea 4: Permuted Block-Sparse Attention (PBS-Attn)
# Instead of fine-grained token sorting, we do chunk-wise / block-wise evaluation.
# We pool queries and keys into blocks, find the top `num_blocks` correlated blocks,
# and gather those tokens. This avoids random memory reads and promotes coalesced GPU loads!

class PBSAttention(BartAttention):
    def __init__(self, base_attn: BartAttention, block_size: int = 32, num_blocks: int = 4):
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
        self.num_blocks = num_blocks

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

        if src_len <= self.block_size * self.num_blocks:
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
            n_blocks_q = max(1, tgt_len // self.block_size)
            n_blocks_k = max(1, src_len // self.block_size)
            
            # Simple average pooling over chunks
            Q_blocks = Q.view(BH, n_blocks_q, self.block_size, self.head_dim).mean(dim=2)
            K_blocks = K.view(BH, n_blocks_k, self.block_size, self.head_dim).mean(dim=2)
            
            # Block-to-block similarity
            block_scores = torch.bmm(Q_blocks, K_blocks.transpose(1, 2))
            
            # Select top-M blocks
            M = min(self.num_blocks, n_blocks_k)
            _, top_blocks = torch.topk(block_scores, k=M, dim=-1)  
            
            # Expand block indices to token indices [BH, tgt_len, M]
            top_blocks_exp = top_blocks.unsqueeze(2).expand(-1, -1, self.block_size, -1).reshape(BH, tgt_len, M)
            
            block_offset = torch.arange(self.block_size, device=Q.device).view(1, 1, 1, self.block_size)
            base_idx = (top_blocks_exp * self.block_size).unsqueeze(-1)  
            token_idx = (base_idx + block_offset).reshape(BH, tgt_len, M * self.block_size) 
            
            M_tokens = M * self.block_size
            idx_gather = token_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            K_sel = torch.gather(K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_gather)
            V_sel = torch.gather(V.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_gather)
            
            scores_sel = torch.matmul(Q.unsqueeze(2), K_sel.transpose(-1, -2)).squeeze(2)
            
            if attention_mask is not None:
                am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if am_bool.dim() == 2: am_bool = am_bool[:, None, None, :]
                am_small = am_bool.expand(bsz, 1, tgt_len, src_len)
                abs_idx_hb = token_idx.view(self.num_heads, bsz, tgt_len, M_tokens)
                allowed_chunks = []
                for h in range(self.num_heads):
                    allowed_h = torch.gather(am_small, -1, abs_idx_hb[h].unsqueeze(1)).squeeze(1)
                    allowed_chunks.append(allowed_h)
                allowed = torch.cat(allowed_chunks, dim=0)
                scores_sel = scores_sel.masked_fill(~allowed, -1e9)
            
            attn_probs = F.softmax(scores_sel, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            out = torch.bmm(
                attn_probs.reshape(BH * tgt_len, 1, M_tokens),
                V_sel.reshape(BH * tgt_len, M_tokens, self.head_dim)
            ).reshape(BH, tgt_len, self.head_dim)

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim) \
                        .transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        if use_cache:
            return (attn_output, None, (key_states, value_states))
        return (attn_output, None)

def patch_bart(model: nn.Module, block_size: int = 32, num_blocks: int = 4):
    def _recurse(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, BartAttention):
                if getattr(child, "is_decoder", False): continue
                setattr(module, name, PBSAttention(child, block_size, num_blocks))
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
    def __init__(self, base_model, block_size=32, num_blocks=4):
        super().__init__()
        self.model = base_model
        patch_bart(self.model, block_size=block_size, num_blocks=num_blocks)
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
