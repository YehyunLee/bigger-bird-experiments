#!/usr/bin/env python3
"""Run a single RULER-style long-context experiment (NIAH / MQ-NIAH).

Probes whether sparse attention preserves retrieval signal at varying context
lengths and needle depths. Reuses the from-scratch LRA encoder + 13 sparse patches.

Usage:
  python run_ruler_experiment.py --task niah --exp 1 --seq 4096 --depth 0.5
  python run_ruler_experiment.py --task mq_niah --exp 7 --seq 8192 --depth 0.9 --size ruler-report
  python run_ruler_experiment.py --list
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

from shared.lra_model import build_lra_model
from shared.ruler_dataset import TASK_INFO, build_ruler_dataset
from shared.runner import TrainConfig, run_lra
from run_experiment import EXPERIMENT_CONFIGS

RULER_COMPUTE = {
    "ruler-smoke": {
        "train_samples": 256,
        "eval_samples": 128,
        "batch_size": 8,
        "grad_accum": 1,
        "epochs": 2,
        "desc": "Quick pipeline sanity check",
    },
    "ruler-oom": {
        "train_samples": 32,
        "eval_samples": 16,
        "batch_size": 1,
        "grad_accum": 1,
        "epochs": 1,
        "desc": "Tiny budget for OOM/survival probes at long context",
    },
    "ruler-report": {
        "train_samples": 1000,
        "eval_samples": 200,
        "batch_size": 4,
        "grad_accum": 2,
        "epochs": 4,
        "desc": "Moderate budget for depth × context retention sweeps",
    },
    "ruler-full": {
        "train_samples": 5000,
        "eval_samples": 500,
        "batch_size": 8,
        "grad_accum": 2,
        "epochs": 8,
        "desc": "Full synthetic training budget",
    },
}

DEFAULT_SEQ = {"niah": 4096, "mq_niah": 4096}
DEFAULT_DEPTH = 0.5


def main():
    parser = argparse.ArgumentParser(
        description="Run a single RULER-style long-context experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=list(TASK_INFO.keys()), help="RULER task")
    parser.add_argument("--exp", type=int, choices=sorted(EXPERIMENT_CONFIGS.keys()),
                        help="Experiment number (0=dense baseline)")
    parser.add_argument("--size", choices=list(RULER_COMPUTE.keys()), default="ruler-smoke",
                        help="Compute preset (default: ruler-smoke)")
    parser.add_argument("--seq", type=int, help="Context window")
    parser.add_argument("--depth", type=float, help="Needle depth fraction 0..1 (default 0.5)")
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--accum", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--list", action="store_true", help="List presets and tasks")
    args = parser.parse_args()

    if args.list:
        print("\nRULER compute presets:")
        for name, c in RULER_COMPUTE.items():
            print(f"  {name}: {c['desc']} "
                  f"({c['train_samples']} train, {c['epochs']} epochs, batch {c['batch_size']}x{c['grad_accum']})")
        print("\nTasks:")
        for t, info in TASK_INFO.items():
            print(f"  {t}: num_labels={info['num_labels']}, default_seq={DEFAULT_SEQ[t]}")
        print("\nExperiments:")
        for num, (name, _, params) in EXPERIMENT_CONFIGS.items():
            print(f"  {num}: {name} ({params})")
        return

    if args.task is None or args.exp is None:
        parser.error("--task and --exp are required (unless --list)")

    compute = dict(RULER_COMPUTE[args.size])
    if args.train_samples:
        compute["train_samples"] = args.train_samples
    if args.eval_samples:
        compute["eval_samples"] = args.eval_samples
    if args.batch:
        compute["batch_size"] = args.batch
    if args.accum:
        compute["grad_accum"] = args.accum
    if args.epochs:
        compute["epochs"] = args.epochs

    seq_len = args.seq or DEFAULT_SEQ[args.task]
    depth = args.depth if args.depth is not None else DEFAULT_DEPTH

    print(f"\n{'='*70}")
    print(f"RULER task: {args.task} | exp {args.exp} | seq {seq_len} | depth {depth} | {args.size}")
    print(f"{'='*70}\n")

    data = build_ruler_dataset(
        task=args.task,
        seq_len=seq_len,
        needle_depth=depth,
        train_samples=compute["train_samples"],
        eval_samples=compute["eval_samples"],
        seed=args.seed,
    )
    ds = {"train": data["train"], "validation": data["validation"]}

    model, exp_name, meta = build_lra_model(
        task=args.task,
        exp_num=args.exp,
        vocab_size=data["vocab_size"],
        seq_len=seq_len,
        num_labels=data["num_labels"],
        pair=data["pair"],
    )

    train_cfg = TrainConfig(
        epochs=compute["epochs"],
        per_device_train_bs=compute["batch_size"],
        per_device_eval_bs=compute["batch_size"],
        grad_accum_steps=compute["grad_accum"],
        lr=args.lr,
        use_cpu=args.cpu,
        torch_compile=args.compile,
    )

    run_lra(
        task=args.task,
        exp_name=exp_name,
        model=model,
        ds=ds,
        cfg=train_cfg,
        num_labels=data["num_labels"],
        seq_len=seq_len,
        vocab_size=data["vocab_size"],
        pair=data["pair"],
        extra_meta={**meta, "compute_preset": args.size, "needle_depth": depth},
        save_weights=args.save_weights,
        track="ruler",
    )


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
