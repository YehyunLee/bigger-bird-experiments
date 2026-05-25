"""Vectorized attention mask builders for NSA kernels."""

from __future__ import annotations

import torch


def build_key_mask(
    attention_mask: torch.Tensor | None,
    bsz: int,
    num_heads: int,
    src_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Per-key validity [BH, src_len] from padding mask [B, src_len]."""
    if attention_mask is None:
        return None
    am = attention_mask if attention_mask.dtype == torch.bool else attention_mask > -1e-8
    if am.dim() == 4:
        am = am[:, 0, 0, :src_len]
    elif am.dim() == 2:
        am = am[:, :src_len]
    return am.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(bsz * num_heads, src_len).to(device)


def build_gather_key_mask(
    attention_mask: torch.Tensor | None,
    bsz: int,
    num_heads: int,
    tgt_len: int,
    token_idx: torch.Tensor,
) -> torch.Tensor | None:
    """Gathered-key validity [BH, T, M] from token indices."""
    if attention_mask is None:
        return None
    am = attention_mask if attention_mask.dtype == torch.bool else attention_mask > -1e-8
    if am.dim() == 4:
        am = am[:, 0, 0, :]
    src_len = am.size(-1)
    bh = token_idx.size(0)
    idx = token_idx.view(bsz, num_heads, tgt_len, -1)
    am_bht = am.unsqueeze(1).unsqueeze(1).expand(bsz, num_heads, tgt_len, src_len)
    allowed = torch.gather(am_bht, -1, idx)
    return allowed.reshape(bh, tgt_len, -1)


def build_block_ok(
    attention_mask: torch.Tensor | None,
    bsz: int,
    num_heads: int,
    n_cmp: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Block validity [BH, n_cmp] from block-start token mask."""
    if attention_mask is None:
        return None
    am = attention_mask if attention_mask.dtype == torch.bool else attention_mask > -1e-8
    if am.dim() == 4:
        am = am[:, 0, 0, :]
    elif am.dim() == 2:
        am = am
    block_starts = torch.arange(n_cmp, device=device) * stride
    block_starts = block_starts.clamp(max=am.size(-1) - 1)
    block_ok = am[:, block_starts]
    return block_ok.unsqueeze(1).expand(bsz, num_heads, n_cmp).reshape(bsz * num_heads, n_cmp)
