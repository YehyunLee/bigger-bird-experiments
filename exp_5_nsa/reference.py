"""PyTorch reference implementations for NSA attention branches (inference, no dropout)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mask_scores_key_mask(
    scores: torch.Tensor,
    key_mask: torch.Tensor | None,
    *,
    per_query_mask: bool,
) -> torch.Tensor:
    if key_mask is None:
        return scores
    if per_query_mask:
        return scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
    # key_mask [BH, S] broadcast to [BH, T, S] via indexing done by caller
    return scores


def ref_sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    key_mask: torch.Tensor | None = None,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """q,k,v: [BH, T, D]. key_mask: optional [BH, S] (1 = valid key)."""
    bh, tgt_len, head_dim = q.shape
    src_len = k.size(1)
    w = min(window_size, src_len)
    device = q.device

    t = torch.arange(tgt_len, device=device)
    starts = torch.clamp(t - w + 1, min=0)
    offsets = torch.arange(w, device=device)
    local_idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(max=src_len - 1)
    local_idx_exp = local_idx.unsqueeze(0).expand(bh, -1, -1)

    idx_gather = local_idx_exp.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    k_sel = torch.gather(
        k.unsqueeze(1).expand(bh, tgt_len, src_len, head_dim), 2, idx_gather
    )
    v_sel = torch.gather(
        v.unsqueeze(1).expand(bh, tgt_len, src_len, head_dim), 2, idx_gather
    )

    scores = torch.matmul(q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2) * scale
    if key_mask is not None:
        km = key_mask if key_mask.dtype == torch.bool else key_mask > 0
        allowed = torch.gather(km.unsqueeze(1).expand(bh, tgt_len, src_len), 2, local_idx_exp)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

    attn = F.softmax(scores, dim=-1)
    return torch.bmm(
        attn.reshape(bh * tgt_len, 1, w),
        v_sel.reshape(bh * tgt_len, w, head_dim),
    ).reshape(bh, tgt_len, head_dim)


def ref_compressed_causal_attention(
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    v_cmp: torch.Tensor,
    block_size: int,
    stride: int,
    block_ok: torch.Tensor | None = None,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    bh, tgt_len, _ = q.shape
    n_cmp = k_cmp.size(1)
    device = q.device

    scores = torch.bmm(q, k_cmp.transpose(1, 2)) * scale
    block_ends = torch.arange(n_cmp, device=device) * stride + block_size
    causal = block_ends.unsqueeze(0) <= torch.arange(tgt_len, device=device).unsqueeze(1)
    scores = scores.masked_fill(~causal.unsqueeze(0), torch.finfo(scores.dtype).min)

    if block_ok is not None:
        ok = block_ok if block_ok.dtype == torch.bool else block_ok > 0
        scores = scores.masked_fill(~ok.unsqueeze(1), torch.finfo(scores.dtype).min)

    attn = F.softmax(scores, dim=-1)
    return torch.bmm(attn, v_cmp)


def ref_sparse_gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_idx: torch.Tensor,
    key_mask: torch.Tensor | None = None,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    bh, tgt_len, dim = q.shape
    m_tokens = token_idx.size(-1)

    idx_gather = token_idx.unsqueeze(-1).expand(-1, -1, -1, dim)
    k_sel = torch.gather(k.unsqueeze(1).expand(bh, tgt_len, k.size(1), dim), 2, idx_gather)
    v_sel = torch.gather(v.unsqueeze(1).expand(bh, tgt_len, v.size(1), dim), 2, idx_gather)

    scores = torch.matmul(q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2) * scale
    if key_mask is not None:
        km = key_mask if key_mask.dtype == torch.bool else key_mask > 0
        scores = scores.masked_fill(~km, torch.finfo(scores.dtype).min)

    attn = F.softmax(scores, dim=-1)
    return torch.bmm(
        attn.reshape(bh * tgt_len, 1, m_tokens),
        v_sel.reshape(bh * tgt_len, m_tokens, dim),
    ).reshape(bh, tgt_len, dim)
