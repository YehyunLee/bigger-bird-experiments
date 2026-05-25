"""NSA Triton kernels (inference-only fused attention)."""

from .common import triton_available
from .compressed import compressed_causal_attention
from .gather import sparse_gather_attention
from .masks import build_block_ok, build_gather_key_mask, build_key_mask
from .window import sliding_window_attention

__all__ = [
    "triton_available",
    "sliding_window_attention",
    "compressed_causal_attention",
    "sparse_gather_attention",
    "build_key_mask",
    "build_gather_key_mask",
    "build_block_ok",
]
