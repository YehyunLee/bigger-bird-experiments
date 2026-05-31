#!/usr/bin/env python3
"""Visualize training trajectories (per-epoch metrics) for each experiment."""

import json
import os
import glob
from typing import List, Dict, Any
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Install matplotlib: pip install matplotlib")
    exit(1)

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")

@dataclass
class TrainingRun:
    exp_name: str
    timestamp: str
    train_samples: int
    epochs: int
    trajectory: List[Dict]  # per-epoch metrics
    final_f1: float
    final_acc: float
    raw: Dict[str, Any]

def load_all_runs() -> List[TrainingRun]:
    """Load all runs with trajectory data."""
    runs = []
    
    exp_dirs = [d for d in os.listdir(BENCHMARK_DIR) 
                if os.path.isdir(os.path.join(BENCHMARK_DIR, d)) and not d.startswith('.')]
    
    for exp_dir in exp_dirs:
        json_files = glob.glob(os.path.join(BENCHMARK_DIR, exp_dir, "eval_*.json"))
        for json_path in json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)
                
                meta = data["experiment_metadata"]
                perf = data["performance_metrics"]
                traj = perf.get("trajectory", [])
                
                if not traj:
                    continue  # Skip old runs without trajectory
                
                runs.append(TrainingRun(
                    exp_name=meta["name"],
                    timestamp=meta["timestamp"],
                    train_samples=meta["dataset_info"]["train_size"],
                    epochs=meta["training_config"]["epochs"],
                    trajectory=traj,
                    final_f1=perf["eval"].get("eval_f1", 0),
                    final_acc=perf["eval"].get("eval_accuracy", 0),
                    raw=data
                ))
            except Exception as e:
                print(f"Warning: Could not load {json_path}: {e}")
    
    return sorted(runs, key=lambda x: (x.exp_name, x.timestamp))

def plot_training_trajectories(runs: List[TrainingRun], save_dir: str):
    """Plot per-epoch training curves."""
    
    colors = {
        'exp_1_deepseek_topk': '#1f77b4',
        'exp_2_lightning_hybrid': '#ff7f0e',
        'exp_3_dynamic_globals': '#2ca02c',
        'exp_4_pbs_attn': '#d62728',
        'exp_5_nsa': '#9467bd',
        'exp_6_s2_hhst': '#8c564b',
    }
    
    # Group by experiment
    by_exp = {}
    for r in runs:
        if r.exp_name not in by_exp:
            by_exp[r.exp_name] = []
        by_exp[r.exp_name].append(r)
    
    # Keep only the latest run per experiment for clarity
    latest_runs = []
    for exp_name, exp_runs in by_exp.items():
        latest = max(exp_runs, key=lambda x: x.timestamp)
        latest_runs.append(latest)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Trajectories (Per-Epoch)', fontsize=16, fontweight='bold')
    
    # 1. F1 over epochs
    ax = axes[0, 0]
    for run in latest_runs:
        epochs = [p["epoch"] for p in run.trajectory if p["epoch"] is not None]
        f1s = [p["eval_f1"] for p in run.trajectory if p["eval_f1"] is not None]
        if epochs and f1s:
            ax.plot(epochs, f1s, 'o-', label=run.exp_name.replace('exp_', ''),
                   color=colors.get(run.exp_name, 'gray'), linewidth=2, markersize=8)
            # Annotate final value
            if len(epochs) > 0:
                ax.annotate(f'{f1s[-1]:.3f}', 
                           xy=(epochs[-1], f1s[-1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score over Training')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy over epochs
    ax = axes[0, 1]
    for run in latest_runs:
        epochs = [p["epoch"] for p in run.trajectory if p["epoch"] is not None]
        accs = [p["eval_accuracy"] for p in run.trajectory if p["eval_accuracy"] is not None]
        if epochs and accs:
            ax.plot(epochs, accs, 's-', label=run.exp_name.replace('exp_', ''),
                   color=colors.get(run.exp_name, 'gray'), linewidth=2, markersize=8)
            if len(epochs) > 0:
                ax.annotate(f'{accs[-1]:.3f}', 
                           xy=(epochs[-1], accs[-1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy over Training')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 3. Train vs Eval Loss
    ax = axes[1, 0]
    for run in latest_runs:
        epochs = [p["epoch"] for p in run.trajectory if p["epoch"] is not None]
        train_losses = [p["train_loss"] for p in run.trajectory if p["train_loss"] is not None]
        eval_losses = [p["eval_loss"] for p in run.trajectory if p["eval_loss"] is not None]
        
        color = colors.get(run.exp_name, 'gray')
        if epochs and train_losses:
            ax.plot(epochs, train_losses, 'o--', label=f"{run.exp_name.replace('exp_', '')} (train)",
                   color=color, alpha=0.5)
        if epochs and eval_losses:
            ax.plot(epochs, eval_losses, 's-', label=f"{run.exp_name.replace('exp_', '')} (eval)",
                   color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves (Train vs Eval)')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 4. Steps vs Metrics (showing actual training progress)
    ax = axes[1, 1]
    for run in latest_runs:
        steps = [p["step"] for p in run.trajectory if p["step"] is not None]
        f1s = [p["eval_f1"] for p in run.trajectory if p["eval_f1"] is not None]
        if steps and f1s:
            # Normalize steps to show progress
            total_steps = max(steps) if steps else 1
            norm_steps = [s / total_steps for s in steps]
            ax.plot(norm_steps, f1s, 'o-', label=run.exp_name.replace('exp_', ''),
                   color=colors.get(run.exp_name, 'gray'), linewidth=2, markersize=8)
    ax.set_xlabel('Training Progress (normalized steps)')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 vs Training Progress')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_trajectories.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory plot saved to: {save_path}")
    
    # Also print table
    print("\n" + "="*80)
    print("TRAINING TRAJECTORY SUMMARY")
    print("="*80)
    for run in latest_runs:
        print(f"\n{run.exp_name} ({run.train_samples} samples, {run.epochs} epochs):")
        print(f"{'Epoch':>6} {'Step':>8} {'Train Loss':>12} {'Eval Loss':>12} {'F1':>8} {'Acc':>8}")
        print("-"*70)
        for p in run.trajectory:
            epoch = p.get("epoch", "-")
            step = p.get("step", "-")
            tl = f"{p['train_loss']:.4f}" if p.get("train_loss") else "-"
            el = f"{p['eval_loss']:.4f}" if p.get("eval_loss") else "-"
            f1 = f"{p['eval_f1']:.3f}" if p.get("eval_f1") else "-"
            acc = f"{p['eval_accuracy']:.3f}" if p.get("eval_accuracy") else "-"
            print(f"{epoch:>6} {step:>8} {tl:>12} {el:>12} {f1:>8} {acc:>8}")
        print(f"Final F1: {run.final_f1:.3f}, Final Acc: {run.final_acc:.3f}")
    print("="*80)

def main():
    runs = load_all_runs()
    
    if not runs:
        print("No runs with trajectory data found.")
        print("Run experiments first with the updated runner (captures per-epoch metrics).")
        return
    
    print(f"Loaded {len(runs)} runs with trajectory data")
    plot_training_trajectories(runs, BENCHMARK_DIR)

if __name__ == "__main__":
    main()
