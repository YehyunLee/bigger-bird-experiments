import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartAttention
from transformers.modeling_outputs import SequenceClassifierOutput

# Idea B: DeepSeek Router + PBS Clustering (HYBRID)
# Step 1: Low-rank Q/K projection -> rough scores (DeepSeek Lightning Indexer)
# Step 2: Block-level pooling of the rough scores -> select top-M most-relevant BLOCKS
# Step 3: Within those blocks, select top-K tokens via the same rough scores
# Step 4: Sort selected indices ascending so memory reads are CONTIGUOUS (PBS coalescing)
# Step 5: Run high-precision softmax attention on the selected K keys.


class DeepSeekPBSAttention(BartAttention):
    def __init__(self, base_attn: BartAttention,
                 top_k: int = 64,
                 low_rank_dim: int = 16,
                 block_size: int = 32,
                 num_blocks: int = 4):
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
        self.block_size = block_size
        self.num_blocks = num_blocks

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                use_cache=False, **kwargs):
        for k in ("cache_position", "position_bias", "alibi_bias"):
            kwargs.pop(k, None)

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

        # Short sequences: full dense
        if src_len <= self.top_k or src_len < self.block_size * 2:
            scores = torch.bmm(Q, K.transpose(1, 2))
            if attention_mask is not None:
                am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if am_bool.dim() == 2: am_bool = am_bool[:, None, None, :]
                me = am_bool.expand(bsz, self.num_heads, tgt_len, src_len).reshape(BH, tgt_len, src_len)
                scores = scores.masked_fill(~me, -1e9)
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.bmm(attn, V).reshape(BH, tgt_len, self.head_dim)
        else:
            # --- Step 1: low-rank rough scores ---
            d_low = min(self.low_rank_dim, self.head_dim)
            Q_low = Q[:, :, :d_low]
            K_low = K[:, :, :d_low]
            rough = torch.bmm(Q_low, K_low.transpose(1, 2)) / (d_low ** 0.5)  # [BH, Tq, Src]

            # Build a clean [B, src_len] token-level mask once
            token_mask = None
            if attention_mask is not None:
                am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
                if am_bool.dim() == 4:
                    token_mask = am_bool[:, 0, 0, :]  # [B, src_len]
                elif am_bool.dim() == 3:
                    token_mask = am_bool[:, 0, :]
                else:
                    token_mask = am_bool  # [B, src_len]
                me = token_mask[:, None, None, :].expand(bsz, self.num_heads, tgt_len, src_len).reshape(BH, tgt_len, src_len)
                rough = rough.masked_fill(~me, -1e9)

            # --- Step 2: block-pool the rough scores -> [BH, Tq, n_blocks] ---
            n_blocks_total = src_len // self.block_size
            usable_src = n_blocks_total * self.block_size
            block_scores = rough[:, :, :usable_src].view(BH, tgt_len, n_blocks_total, self.block_size).mean(dim=-1)

            # --- Step 3: pick top-M blocks per query ---
            M_blocks = min(self.num_blocks, n_blocks_total)
            _, top_blk = torch.topk(block_scores, k=M_blocks, dim=-1)  # [BH, Tq, M_blocks]

            # --- Step 3b: within those blocks, take top-(top_k/M_blocks) per block ---
            per_block_k = max(1, self.top_k // M_blocks)
            # Build mask: only block positions are valid
            blk_starts = top_blk * self.block_size  # [BH, Tq, M_blocks]
            offs = torch.arange(self.block_size, device=Q.device)
            blk_tokens = blk_starts.unsqueeze(-1) + offs  # [BH, Tq, M_blocks, block_size]
            blk_tokens_flat = blk_tokens.reshape(BH, tgt_len, M_blocks * self.block_size)
            # Gather rough scores for those tokens
            blk_rough = torch.gather(rough, 2, blk_tokens_flat)  # [BH, Tq, M_blocks*block_size]
            K_target = min(self.top_k, blk_tokens_flat.size(-1))
            _, sel_rel = torch.topk(blk_rough, k=K_target, dim=-1)
            top_idx = torch.gather(blk_tokens_flat, 2, sel_rel)  # absolute idx [BH, Tq, K_target]

            # --- Step 4: sort indices for coalesced memory reads (PBS trick) ---
            top_idx, _ = torch.sort(top_idx, dim=-1)

            # --- Step 5: high-precision attention on selected keys ---
            idx_g = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            K_sel = torch.gather(K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_g)
            V_sel = torch.gather(V.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_g)
            scores = torch.matmul(Q.unsqueeze(2), K_sel.transpose(-1, -2)).squeeze(2)

            if token_mask is not None:
                am_bh = token_mask[:, None, :].expand(bsz, self.num_heads, src_len).reshape(BH, src_len)
                allowed = torch.gather(am_bh.unsqueeze(1).expand(BH, tgt_len, src_len), 2, top_idx)
                scores = scores.masked_fill(~allowed, -1e9)

            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.bmm(
                attn.reshape(BH * tgt_len, 1, K_target),
                V_sel.reshape(BH * tgt_len, K_target, self.head_dim)
            ).reshape(BH, tgt_len, self.head_dim)

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        return (self.out_proj(attn_output), None)


def patch_bart(model: nn.Module, **kw):
    def _rec(m):
        for n, c in list(m.named_children()):
            if isinstance(c, BartAttention):
                if getattr(c, "is_decoder", False): continue
                setattr(m, n, DeepSeekPBSAttention(c, **kw))
            else:
                _rec(c)
    _rec(model)


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
    def __init__(self, base_model, top_k=64, low_rank_dim=16, block_size=32, num_blocks=4):
        super().__init__()
        self.model = base_model
        patch_bart(self.model, top_k=top_k, low_rank_dim=low_rank_dim,
                   block_size=block_size, num_blocks=num_blocks)
        hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model")
        self.attn_pool = AttnPool(hidden_size)

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
