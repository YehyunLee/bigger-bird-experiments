"""Dispatch helpers for Triton vs PyTorch fallback."""

from __future__ import annotations

import torch

from .common import triton_available


def should_use_triton(use_triton: bool, q: torch.Tensor, *, training: bool = False) -> bool:
    """True when fused Triton kernels may be used for tensor q (inference-only)."""
    return bool(use_triton and not training and q.is_cuda and triton_available())
