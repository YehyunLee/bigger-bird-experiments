"""Shared Triton attention kernels (inference-only fused attention)."""

from .band import band_attention
from .common import (
    MODE_BAND,
    MODE_COMPRESSED,
    MODE_GATHER,
    MODE_WINDOW,
    triton_available,
)
from .dispatch import should_use_triton
from .gather import sparse_gather_attention
from .linear import elu_linear_attention
from .masks import build_gather_key_mask, build_key_mask
from .window import sliding_window_attention

__all__ = [
    "MODE_BAND",
    "MODE_COMPRESSED",
    "MODE_GATHER",
    "MODE_WINDOW",
    "triton_available",
    "should_use_triton",
    "sliding_window_attention",
    "sparse_gather_attention",
    "band_attention",
    "elu_linear_attention",
    "build_key_mask",
    "build_gather_key_mask",
]
