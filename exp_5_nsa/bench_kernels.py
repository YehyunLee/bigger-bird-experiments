#!/usr/bin/env python3
"""Micro-benchmark: PyTorch gather path vs Triton kernels."""

from __future__ import annotations

import math
import time

import torch

from exp_5_nsa.kernels import (
    build_block_ok,
    build_gather_key_mask,
    build_key_mask,
    compressed_causal_attention,
    sliding_window_attention,
    sparse_gather_attention,
    triton_available,
)
from exp_5_nsa.reference import (
    ref_compressed_causal_attention,
    ref_sliding_window_attention,
    ref_sparse_gather_attention,
)

HEAD_DIM = 64
BLOCK_SIZE = 32
STRIDE = 32
TOPK_BLOCKS = 4
WINDOW_SIZE = 128


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn, warmup: int = 5, reps: int = 20) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync()
    return (time.perf_counter() - t0) / reps * 1000.0


def main():
    if not torch.cuda.is_available():
        print("CUDA required for benchmarks")
        return
    if not triton_available():
        print("Triton not available; install python3-dev and use CUDA PyTorch")

    device = "cuda"
    dtype = torch.float16
    bsz, num_heads = 4, 12
    bh = bsz * num_heads
    tgt_len, src_len = 512, 768
    gen = torch.Generator(device=device).manual_seed(0)

    q = torch.randn(bh, tgt_len, HEAD_DIM, generator=gen, device=device, dtype=dtype)
    k = torch.randn(bh, src_len, HEAD_DIM, generator=gen, device=device, dtype=dtype)
    v = torch.randn(bh, src_len, HEAD_DIM, generator=gen, device=device, dtype=dtype)

    w = min(WINDOW_SIZE, src_len)
    am = torch.ones(bsz, src_len, dtype=torch.bool, device=device)
    am[:, src_len - 50 :] = False
    key_mask = build_key_mask(am, bsz, num_heads, src_len, device)

    pad = (BLOCK_SIZE - src_len % BLOCK_SIZE) % BLOCK_SIZE
    k_pad = torch.nn.functional.pad(k, (0, 0, 0, pad))
    v_pad = torch.nn.functional.pad(v, (0, 0, 0, pad))
    n_cmp = k_pad.size(1) // BLOCK_SIZE
    k_cmp = k_pad.view(bh, n_cmp, BLOCK_SIZE, HEAD_DIM).mean(dim=2)
    v_cmp = v_pad.view(bh, n_cmp, BLOCK_SIZE, HEAD_DIM).mean(dim=2)
    block_ok = build_block_ok(am, bsz, num_heads, n_cmp, STRIDE, device)

    n_blocks = math.ceil(src_len / BLOCK_SIZE)
    m = min(TOPK_BLOCKS, n_blocks)
    top_blocks = torch.randint(0, n_blocks, (bh, tgt_len, m), device=device)
    block_offset = torch.arange(BLOCK_SIZE, device=device).view(1, 1, 1, BLOCK_SIZE)
    token_idx = (
        (top_blocks.unsqueeze(2) * BLOCK_SIZE).unsqueeze(-1) + block_offset
    ).reshape(bh, tgt_len, m * BLOCK_SIZE).clamp(max=src_len - 1)
    gather_mask = build_gather_key_mask(am, bsz, num_heads, tgt_len, token_idx)

    print(f"Shapes: BH={bh}, T={tgt_len}, S={src_len}, dtype={dtype}")
    print(f"Triton available: {triton_available()}\n")

    rows = [
        (
            "window",
            lambda: ref_sliding_window_attention(
                q.float(), k.float(), v.float(), w, key_mask, scale=1.0
            ),
            lambda: sliding_window_attention(q, k, v, w, key_mask, scale=1.0),
        ),
        (
            "compressed",
            lambda: ref_compressed_causal_attention(
                q.float(), k_cmp.float(), v_cmp.float(), BLOCK_SIZE, STRIDE, block_ok, scale=1.0
            ),
            lambda: compressed_causal_attention(
                q, k_cmp, v_cmp, BLOCK_SIZE, STRIDE, block_ok, scale=1.0
            ),
        ),
        (
            "gather",
            lambda: ref_sparse_gather_attention(
                q.float(), k.float(), v.float(), token_idx, gather_mask, scale=1.0
            ),
            lambda: sparse_gather_attention(q, k, v, token_idx, gather_mask, scale=1.0),
        ),
    ]

    for name, torch_fn, triton_fn in rows:
        torch.cuda.reset_peak_memory_stats()
        ms_ref = _bench(torch_fn)
        mem_ref = torch.cuda.max_memory_allocated() / 1e6

        torch.cuda.reset_peak_memory_stats()
        ms_tri = _bench(triton_fn)
        mem_tri = torch.cuda.max_memory_allocated() / 1e6

        speedup = ms_ref / ms_tri if ms_tri > 0 else float("inf")
        print(
            f"{name:12}  pytorch {ms_ref:7.2f} ms  ({mem_ref:6.1f} MB peak)  |  "
            f"triton {ms_tri:7.2f} ms  ({mem_tri:6.1f} MB peak)  |  {speedup:.2f}x"
        )


if __name__ == "__main__":
    main()
