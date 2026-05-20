#!/usr/bin/env python3
"""
Empirical complexity verification.
Micro-benchmarks the attention forward pass directly to verify O(n) vs O(n²).

Usage:
  python benchmarks/complexity_verify.py --exp 1,2,3,4
  python benchmarks/complexity_verify.py --exp 0 --seq 128,256,512,1024,2048,4096
"""

import sys
import os
import argparse
import json
import time
import gc
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.bart.modeling_bart import BartAttention

from exp_1_deepseek_topk.model import DeepSeekTopKAttention
from exp_2_lightning_hybrid.model import LightningHybridAttention
from exp_3_dynamic_globals.model import DynamicGlobalAttention
from exp_4_pbs_attn.model import PBSAttention


class DummyBartAttention(BartAttention):
    """Minimal BART attention for timing baseline."""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__(embed_dim, num_heads, dropout, is_decoder=False, bias=True)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, **kwargs):
        bsz, tgt_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states) * (self.head_dim ** -0.5)
        key = self._shape(self.k_proj(hidden_states), -1, bsz)
        value = self._shape(self.v_proj(hidden_states), -1, bsz)
        Q = self._shape(query, tgt_len, bsz).reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        K = key.reshape(bsz * self.num_heads, -1, self.head_dim)
        V = value.reshape(bsz * self.num_heads, -1, self.head_dim)
        scores = torch.bmm(Q, K.transpose(1, 2))
        attn = nn.functional.softmax(scores, dim=-1)
        out = torch.bmm(attn, V)
        out = out.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        return self.out_proj(out), None


def time_attention(module, batch_size, seq_len, device, n_trials=20, warmup=5):
    """Time a single attention forward pass. Returns avg ms."""
    hidden = torch.randn(batch_size, seq_len, 768, device=device)
    module = module.to(device).eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(hidden)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(n_trials):
                _ = module(hidden)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_trials  # ms per forward
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_trials):
                _ = module(hidden)
        return (time.perf_counter() - t0) * 1000 / n_trials


def create_module(exp_num, device):
    """Create an attention module for the given experiment."""
    base = DummyBartAttention(embed_dim=768, num_heads=12, dropout=0.1).to(device)
    if exp_num == 0:
        return base
    elif exp_num == 1:
        return DeepSeekTopKAttention(base, top_k=64, low_rank_dim=16)
    elif exp_num == 2:
        return LightningHybridAttention(base, block_size=128)
    elif exp_num == 3:
        return DynamicGlobalAttention(base, window_size=64, num_globals=16)
    elif exp_num == 4:
        return PBSAttention(base, block_size=64, num_blocks=2)
    else:
        raise ValueError(f"Unknown exp_num: {exp_num}")


def main():
    parser = argparse.ArgumentParser(description="Empirical complexity verification")
    parser.add_argument("--exp", type=str, default="0,1,2,3,4",
                        help="Comma-separated experiment numbers")
    parser.add_argument("--seq", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size for timing")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of timing trials")
    args = parser.parse_args()

    exp_nums = [int(x.strip()) for x in args.exp.split(",")]
    seq_lengths = [int(x.strip()) for x in args.seq.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Experiments: {exp_nums}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Batch size: {args.batch}, Trials: {args.trials}")

    results = []
    for exp_num in exp_nums:
        exp_name = f"exp_{exp_num}_baseline" if exp_num == 0 else f"exp_{exp_num}"
        print(f"\n--- Timing {exp_name} ---")
        for seq_len in seq_lengths:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            try:
                module = create_module(exp_num, device)
                ms = time_attention(module, args.batch, seq_len, device, n_trials=args.trials)
                print(f"  seq={seq_len:>5} | {ms:>8.3f} ms")
                results.append({
                    "exp_num": exp_num,
                    "exp_name": exp_name,
                    "seq_length": seq_len,
                    "batch_size": args.batch,
                    "time_ms": ms,
                    "oom": False,
                })
            except torch.cuda.OutOfMemoryError:
                print(f"  seq={seq_len:>5} | OOM")
                results.append({
                    "exp_num": exp_num,
                    "exp_name": exp_name,
                    "seq_length": seq_len,
                    "batch_size": args.batch,
                    "time_ms": None,
                    "oom": True,
                })
            except Exception as e:
                print(f"  seq={seq_len:>5} | ERROR: {e}")
                results.append({
                    "exp_num": exp_num,
                    "exp_name": exp_name,
                    "seq_length": seq_len,
                    "batch_size": args.batch,
                    "time_ms": None,
                    "oom": False,
                    "error": str(e),
                })

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "complexity_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": str(device),
                "batch_size": args.batch,
                "trials": args.trials,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nSaved results to: {out_path}")

    # Compute log-log slopes (empirical complexity)
    print("\nEMPIRICAL COMPLEXITY (log-log slope ≈ 1 means O(n), ≈ 2 means O(n²))")
    print("-" * 60)
    from math import log
    for exp_num in exp_nums:
        exp_name = f"exp_{exp_num}_baseline" if exp_num == 0 else f"exp_{exp_num}"
        pts = [(r["seq_length"], r["time_ms"]) for r in results
               if r["exp_num"] == exp_num and r["time_ms"] is not None and not r.get("oom", False)]
        pts = sorted(set(pts))
        if len(pts) >= 2:
            slopes = []
            for i in range(1, len(pts)):
                s1, t1 = pts[i-1]
                s2, t2 = pts[i]
                slope = log(t2 / t1) / log(s2 / s1)
                slopes.append(slope)
            avg_slope = sum(slopes) / len(slopes)
            print(f"  {exp_name:<20} avg slope = {avg_slope:.3f}")
        else:
            print(f"  {exp_name:<20} insufficient data")


if __name__ == "__main__":
    main()
