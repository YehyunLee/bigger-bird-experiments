#!/usr/bin/env python3
"""
Unified experiment runner with flexible compute configs.
Usage:
  python run_experiment.py --exp 3 --size small    # Quick test (1GB mem)
  python run_experiment.py --exp 3 --size medium   # Good GPU (8GB mem)  
  python run_experiment.py --exp 3 --size big      # Big GPU (24GB+ mem)
  python run_experiment.py --exp 3 --size big --epochs 5 --batch 4
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shared.dataset import build_imdb_dataset, DataConfig
from shared.runner import run_experiment, TrainConfig

# Import all experiment models
from exp_1_deepseek_topk.model import PatchedModel as DeepSeekModel
from exp_2_lightning_hybrid.model import PatchedModel as LightningModel
from exp_3_dynamic_globals.model import PatchedModel as DynamicGlobalsModel
from exp_4_pbs_attn.model import PatchedModel as PBSModel

# Compute presets
COMPUTE_CONFIGS = {
    "small": {
        "train_samples": 500,
        "eval_samples": 100,
        "max_length": 256,
        "batch_size": 1,
        "grad_accum": 1,
        "epochs": 2,
        "desc": "Quick test / 8GB RAM / Fast iteration"
    },
    "medium": {
        "train_samples": 2000,
        "eval_samples": 400,
        "max_length": 512,
        "batch_size": 2,
        "grad_accum": 4,
        "epochs": 3,
        "desc": "Good GPU / 16GB VRAM / Solid training"
    },
    "big": {
        "train_samples": 6000,
        "eval_samples": 1000,
        "max_length": 768,
        "batch_size": 4,
        "grad_accum": 8,
        "epochs": 3,
        "desc": "Big GPU / 24GB+ VRAM / Full training"
    },
    "xl": {
        "train_samples": 25000,
        "eval_samples": 2500,
        "max_length": 1024,
        "batch_size": 8,
        "grad_accum": 16,
        "epochs": 5,
        "desc": "Full IMDb / Large GPU / Production"
    }
}

EXPERIMENT_CONFIGS = {
    0: ("exp_0_baseline", None, {"attention": "full_dense"}),
    1: ("exp_1_deepseek_topk", DeepSeekModel, {"top_k": 64, "low_rank_dim": 16}),
    2: ("exp_2_lightning_hybrid", LightningModel, {"block_size": 128}),
    3: ("exp_3_dynamic_globals", DynamicGlobalsModel, {"window_size": 64, "num_globals": 16}),
    4: ("exp_4_pbs_attn", PBSModel, {"block_size": 64, "num_blocks": 2}),
}

def main():
    parser = argparse.ArgumentParser(
        description="Run Bigger Bird experiments with flexible compute configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python run_experiment.py --exp 3 --size small
  
  # Full training on GPU  
  python run_experiment.py --exp 3 --size big
  
  # Custom override
  python run_experiment.py --exp 3 --size medium --epochs 5 --seq 768
  
  # List available configs
  python run_experiment.py --list
        """
    )
    
    parser.add_argument("--exp", type=int, choices=[0,1,2,3,4], required=True,
                       help="Experiment number (0=baseline, 1-4=sparse methods)")
    parser.add_argument("--size", type=str, choices=["small", "medium", "big", "xl"],
                       help="Compute size preset")
    parser.add_argument("--list", action="store_true",
                       help="List available compute presets")
    
    # Override flags
    parser.add_argument("--train-samples", type=int, help="Override training samples")
    parser.add_argument("--eval-samples", type=int, help="Override eval samples")
    parser.add_argument("--seq", type=int, help="Override sequence length")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--accum", type=int, help="Override gradient accumulation")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--grad-checkpoint", action="store_true",
                       help="Enable gradient checkpointing (saves memory)")
    parser.add_argument("--save-weights", action="store_true",
                       help="Save model weights to benchmarks/<exp>/weights_<timestamp>/ for later eval")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable compute presets:")
        print("=" * 70)
        for name, cfg in COMPUTE_CONFIGS.items():
            print(f"\n{name.upper()}: {cfg['desc']}")
            print(f"  Samples: {cfg['train_samples']} train / {cfg['eval_samples']} eval")
            print(f"  Seq length: {cfg['max_length']}")
            print(f"  Batch: {cfg['batch_size']} x {cfg['grad_accum']} accum = {cfg['batch_size'] * cfg['grad_accum']} effective")
            print(f"  Epochs: {cfg['epochs']}")
        print("=" * 70)
        print("\nExperiments:")
        for num, (name, _, params) in EXPERIMENT_CONFIGS.items():
            label = " [BASELINE]" if num == 0 else ""
            print(f"  {num}: {name}{label} ({params})")
        return
    
    if not args.size:
        parser.error("--size is required (unless using --list)")
    
    # Get compute config
    compute = COMPUTE_CONFIGS[args.size].copy()
    
    # Apply overrides
    if args.train_samples: compute["train_samples"] = args.train_samples
    if args.eval_samples: compute["eval_samples"] = args.eval_samples
    if args.seq: compute["max_length"] = args.seq
    if args.batch: compute["batch_size"] = args.batch
    if args.accum: compute["grad_accum"] = args.accum
    if args.epochs: compute["epochs"] = args.epochs
    
    # Get experiment config
    exp_name, ModelClass, model_params = EXPERIMENT_CONFIGS[args.exp]
    
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}" + (" [BASELINE - full dense attention]" if args.exp == 0 else ""))
    print(f"Compute: {args.size.upper()} - {compute['desc']}")
    print(f"Config: {compute['train_samples']} samples, seq={compute['max_length']}, "
          f"batch={compute['batch_size']}x{compute['grad_accum']}, {compute['epochs']} epochs")
    print(f"{'='*70}\n")
    
    # Setup
    model_name = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    base_model.config.classifier_dropout = 0.1
    
    # Enable gradient checkpointing if requested
    if args.grad_checkpoint:
        base_model.gradient_checkpointing_enable()
        print("Gradient checkpointing: ENABLED (saves ~30% memory)\n")
    
    # Baseline: use unpatched model directly
    if ModelClass is None:
        model = base_model
    else:
        model = ModelClass(base_model, **model_params)
    
    # Build dataset
    data_cfg = DataConfig(
        train_samples=compute["train_samples"],
        eval_samples=compute["eval_samples"],
        max_length=compute["max_length"]
    )
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=None)
    
    # Training config
    train_cfg = TrainConfig(
        epochs=compute["epochs"],
        per_device_train_bs=compute["batch_size"],
        per_device_eval_bs=compute["batch_size"],
        grad_accum_steps=compute["grad_accum"],
        lr=args.lr
    )
    
    # Run
    run_experiment(
        exp_name,
        model,
        tokenizer,
        ds,
        train_cfg,
        extra_meta={
            **model_params,
            "compute_preset": args.size,
            "train_samples": compute["train_samples"],
            "seq_length": compute["max_length"]
        },
        save_weights=args.save_weights,
    )

if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
