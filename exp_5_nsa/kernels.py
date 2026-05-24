"""Triton fused kernels for NSA inference (forward-only)."""

from __future__ import annotations

import os
import sysconfig

import torch

_TRITON_IMPORTED = False
_TRITON_READY: bool | None = None

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORTED = True
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore


def _has_python_dev_headers() -> bool:
    include = sysconfig.get_path("include")
    return os.path.isfile(os.path.join(include, "Python.h"))


def triton_available() -> bool:
    """True only if Triton can compile and run on the current CUDA device."""
    global _TRITON_READY
    if _TRITON_READY is not None:
        return _TRITON_READY
    if not _TRITON_IMPORTED or not torch.cuda.is_available() or not _has_python_dev_headers():
        _TRITON_READY = False
        return False
    try:
        _probe_triton()
        _TRITON_READY = True
    except Exception:
        _TRITON_READY = False
    return _TRITON_READY


def _probe_triton() -> None:
    """Compile and run a minimal kernel to verify the Triton toolchain."""
    q = torch.zeros(1, 1, 8, device="cuda", dtype=torch.float16)
    k = torch.zeros(1, 8, 8, device="cuda", dtype=torch.float16)
    v = torch.zeros(1, 8, 8, device="cuda", dtype=torch.float16)
    _launch_sliding_window(q, k, v, 4, None, 0.125)


def _ceil_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


if _TRITON_IMPORTED:

    @triton.jit
    def _sparse_attn_row_kernel(
        Q,
        K,
        V,
        Out,
        Idx,
        Mask,  # optional int8 mask [BH, T, M], 1 = keep
        stride_qb,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_ot,
        stride_od,
        stride_ib,
        stride_it,
        stride_im,
        stride_mb,
        stride_mt,
        stride_mm,
        seq_len,
        n_keys: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_MASK: tl.constexpr,
        SCALE,
    ):
        bh = tl.program_id(0)
        t = tl.program_id(1)

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < head_dim

        q_ptr = Q + bh * stride_qb + t * stride_qt + d_offs * stride_qd
        q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for m in range(n_keys):
            idx = tl.load(Idx + bh * stride_ib + t * stride_it + m * stride_im).to(tl.int32)
            idx = tl.minimum(tl.maximum(idx, 0), seq_len - 1)

            k_ptr = K + bh * stride_kb + idx * stride_ks + d_offs * stride_kd
            v_ptr = V + bh * stride_vb + idx * stride_vs + d_offs * stride_vd
            k = tl.load(k_ptr, mask=d_mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptr, mask=d_mask, other=0.0).to(tl.float32)

            s = tl.sum(q * k, axis=0) * SCALE
            if HAS_MASK:
                keep = tl.load(Mask + bh * stride_mb + t * stride_mt + m * stride_mm).to(tl.int1)
                s = tl.where(keep, s, -float("inf"))

            m_new = tl.maximum(m_i, s)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + p
            acc = acc * alpha + p * v
            m_i = m_new

        out = acc / l_i
        o_ptr = Out + bh * stride_ob + t * stride_ot + d_offs * stride_od
        tl.store(o_ptr, out, mask=d_mask)

    @triton.jit
    def _sliding_window_row_kernel(
        Q,
        K,
        V,
        Out,
        KeyMask,  # optional [BH, seq_len] int8
        stride_qb,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_ot,
        stride_od,
        stride_kmb,
        stride_kms,
        seq_len,
        window_size: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_KMASK: tl.constexpr,
        SCALE,
    ):
        bh = tl.program_id(0)
        t = tl.program_id(1)

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < head_dim

        q_ptr = Q + bh * stride_qb + t * stride_qt + d_offs * stride_qd
        q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)

        start = tl.maximum(0, t - window_size + 1)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for m in range(window_size):
            idx = start + m
            valid = idx <= t
            idx = tl.minimum(idx, seq_len - 1)

            k_ptr = K + bh * stride_kb + idx * stride_ks + d_offs * stride_kd
            v_ptr = V + bh * stride_vb + idx * stride_vs + d_offs * stride_vd
            k = tl.load(k_ptr, mask=d_mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptr, mask=d_mask, other=0.0).to(tl.float32)

            s = tl.sum(q * k, axis=0) * SCALE
            s = tl.where(valid, s, -float("inf"))
            if HAS_KMASK:
                keep = tl.load(KeyMask + bh * stride_kmb + idx * stride_kms).to(tl.int1)
                s = tl.where(keep, s, -float("inf"))

            m_new = tl.maximum(m_i, s)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + p
            acc = acc * alpha + p * v
            m_i = m_new

        out = acc / l_i
        o_ptr = Out + bh * stride_ob + t * stride_ot + d_offs * stride_od
        tl.store(o_ptr, out, mask=d_mask)

    @triton.jit
    def _compressed_causal_row_kernel(
        Q,
        Kc,
        Vc,
        Out,
        BlockOk,  # optional [BH, n_cmp] int8
        stride_qb,
        stride_qt,
        stride_qd,
        stride_kcb,
        stride_kcs,
        stride_kcd,
        stride_vcb,
        stride_vcs,
        stride_vcd,
        stride_ob,
        stride_ot,
        stride_od,
        stride_bob,
        stride_boc,
        n_cmp: tl.constexpr,
        block_size: tl.constexpr,
        stride: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_BMASK: tl.constexpr,
        SCALE,
    ):
        bh = tl.program_id(0)
        t = tl.program_id(1)

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < head_dim

        q_ptr = Q + bh * stride_qb + t * stride_qt + d_offs * stride_qd
        q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for c in range(n_cmp):
            block_end = c * stride + block_size
            causal = block_end <= t

            k_ptr = Kc + bh * stride_kcb + c * stride_kcs + d_offs * stride_kcd
            v_ptr = Vc + bh * stride_vcb + c * stride_vcs + d_offs * stride_vcd
            k = tl.load(k_ptr, mask=d_mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptr, mask=d_mask, other=0.0).to(tl.float32)

            s = tl.sum(q * k, axis=0) * SCALE
            s = tl.where(causal, s, -float("inf"))
            if HAS_BMASK:
                keep = tl.load(BlockOk + bh * stride_bob + c * stride_boc).to(tl.int1)
                s = tl.where(keep, s, -float("inf"))

            m_new = tl.maximum(m_i, s)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + p
            acc = acc * alpha + p * v
            m_i = m_new

        out = acc / l_i
        o_ptr = Out + bh * stride_ob + t * stride_ot + d_offs * stride_od
        tl.store(o_ptr, out, mask=d_mask)


def _launch_sparse_gather(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    key_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    bh, tgt_len, head_dim = q.shape
    n_keys = indices.size(-1)
    out = torch.empty_like(q)
    block_d = _ceil_pow2(head_dim)
    has_mask = key_mask is not None
    mask_i8 = key_mask.to(torch.int8) if has_mask else q  # dummy

    grid = (bh, tgt_len)
    _sparse_attn_row_kernel[grid](
        q,
        k,
        v,
        out,
        indices,
        mask_i8,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1),
        indices.stride(2),
        mask_i8.stride(0) if has_mask else 0,
        mask_i8.stride(1) if has_mask else 0,
        mask_i8.stride(2) if has_mask else 0,
        k.size(1),
        n_keys=n_keys,
        head_dim=head_dim,
        BLOCK_D=block_d,
        HAS_MASK=has_mask,
        SCALE=scale,
        num_warps=4,
    )
    return out


def _launch_sliding_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    key_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    bh, tgt_len, head_dim = q.shape
    seq_len = k.size(1)
    w = min(window_size, seq_len)
    out = torch.empty_like(q)
    block_d = _ceil_pow2(head_dim)
    has_kmask = key_mask is not None
    km = key_mask.to(torch.int8) if has_kmask else q

    grid = (bh, tgt_len)
    _sliding_window_row_kernel[grid](
        q,
        k,
        v,
        out,
        km,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        km.stride(0) if has_kmask else 0,
        km.stride(1) if has_kmask else 0,
        seq_len,
        window_size=w,
        head_dim=head_dim,
        BLOCK_D=block_d,
        HAS_KMASK=has_kmask,
        SCALE=scale,
        num_warps=4,
    )
    return out


def _launch_compressed_causal(
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    v_cmp: torch.Tensor,
    block_size: int,
    stride: int,
    block_ok: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    bh, tgt_len, head_dim = q.shape
    n_cmp = k_cmp.size(1)
    out = torch.empty_like(q)
    block_d = _ceil_pow2(head_dim)
    has_bmask = block_ok is not None
    bok = block_ok.to(torch.int8) if has_bmask else q

    grid = (bh, tgt_len)
    _compressed_causal_row_kernel[grid](
        q,
        k_cmp,
        v_cmp,
        out,
        bok,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cmp.stride(0),
        k_cmp.stride(1),
        k_cmp.stride(2),
        v_cmp.stride(0),
        v_cmp.stride(1),
        v_cmp.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        bok.stride(0) if has_bmask else 0,
        bok.stride(1) if has_bmask else 0,
        n_cmp=n_cmp,
        block_size=block_size,
        stride=stride,
        head_dim=head_dim,
        BLOCK_D=block_d,
        HAS_BMASK=has_bmask,
        SCALE=scale,
        num_warps=4,
    )
    return out


@torch.inference_mode()
def sparse_gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    key_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Fused attention over gathered keys. Shapes: q/k/v [BH,T,D], indices [BH,T,M]."""
    if not triton_available():
        raise RuntimeError("Triton CUDA kernels are not available")
    scale = scale if scale is not None else (q.size(-1) ** -0.5)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    indices = indices.contiguous().to(torch.int32)
    return _launch_sparse_gather(q, k, v, indices, key_mask, scale)


@torch.inference_mode()
def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    key_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal sliding-window attention without materializing gathers."""
    if not triton_available():
        raise RuntimeError("Triton CUDA kernels are not available")
    scale = scale if scale is not None else (q.size(-1) ** -0.5)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    return _launch_sliding_window(q, k, v, window_size, key_mask, scale)


@torch.inference_mode()
def compressed_causal_attention(
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    v_cmp: torch.Tensor,
    block_size: int,
    stride: int,
    block_ok: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Attention over compressed keys with per-query causal block mask."""
    if not triton_available():
        raise RuntimeError("Triton CUDA kernels are not available")
    scale = scale if scale is not None else (q.size(-1) ** -0.5)
    q, k_cmp, v_cmp = q.contiguous(), k_cmp.contiguous(), v_cmp.contiguous()
    return _launch_compressed_causal(q, k_cmp, v_cmp, block_size, stride, block_ok, scale)


def build_key_mask(attention_mask, bsz: int, num_heads: int, src_len: int, device) -> torch.Tensor | None:
    """Per-key validity [BH, src_len] from padding mask [B, src_len]."""
    if attention_mask is None:
        return None
    am = attention_mask if attention_mask.dtype == torch.bool else attention_mask > -1e-8
    if am.dim() == 4:
        am = am[:, 0, 0, :]
    return am.unsqueeze(1).expand(bsz, num_heads, src_len).reshape(bsz * num_heads, src_len).to(device)


def build_gather_key_mask(attention_mask, bsz: int, num_heads: int, tgt_len: int, token_idx: torch.Tensor) -> torch.Tensor | None:
    """Gathered-key validity [BH, T, M] from token indices."""
    if attention_mask is None:
        return None
    am = attention_mask if attention_mask.dtype == torch.bool else attention_mask > -1e-8
    if am.dim() == 2:
        am = am[:, None, None, :]
    src_len = am.size(-1)
    bh = token_idx.size(0)
    am_small = am.expand(bsz, 1, tgt_len, src_len)
    abs_idx_hb = token_idx.view(num_heads, bsz, tgt_len, -1)
    allowed = []
    for h in range(num_heads):
        allowed.append(torch.gather(am_small, -1, abs_idx_hb[h].unsqueeze(1)).squeeze(1))
    return torch.cat(allowed, dim=0)


def build_block_ok(attention_mask, bsz: int, num_heads: int, n_cmp: int, stride: int, device) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    am = attention_mask if attention_mask.dtype == torch.bool else attention_mask > -1e-8
    if am.dim() == 2:
        am = am[:, None, None, :]
    block_starts = torch.arange(n_cmp, device=device) * stride
    block_starts = block_starts.clamp(max=am.size(-1) - 1)
    block_ok = am[:, 0, 0, block_starts]
    return block_ok.unsqueeze(1).expand(bsz, num_heads, n_cmp).reshape(bsz * num_heads, n_cmp)