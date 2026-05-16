#!/usr/bin/env python3
"""Compare experiment results across all benchmark runs."""

import json
import os
import glob
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")

@dataclass
class ExperimentResult:
    name: str
    timestamp: str
    f1: float
    accuracy: float
    train_time: float
    eval_time: float
    train_samples: int
    eval_samples: int
    epochs: int
    train_loss: float
    eval_loss: float
    raw: Dict[str, Any]

def load_all_results() -> List[ExperimentResult]:
    """Load all experiment results from benchmark JSON files."""
    results = []
    
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
                
                results.append(ExperimentResult(
                    name=meta["name"],
                    timestamp=meta["timestamp"],
                    f1=perf["eval"].get("eval_f1", 0),
                    accuracy=perf["eval"].get("eval_accuracy", 0),
                    train_time=perf["training_time_seconds"],
                    eval_time=perf["eval"].get("eval_runtime", 0),
                    train_samples=meta["dataset_info"]["train_size"],
                    eval_samples=meta["dataset_info"]["eval_size"],
                    epochs=meta["training_config"]["epochs"],
                    train_loss=perf["train"].get("train_loss", 0),
                    eval_loss=perf["eval"].get("eval_loss", 0),
                    raw=data
                ))
            except Exception as e:
                print(f"Warning: Could not load {json_path}: {e}")
    
    return sorted(results, key=lambda x: (x.name, x.timestamp))

def group_by_sample_size(results: List[ExperimentResult]) -> Dict[int, Dict[str, ExperimentResult]]:
    """Group results by train_samples. Within each group, keep latest run per experiment."""
    groups: Dict[int, Dict[str, ExperimentResult]] = {}
    for r in results:
        n = r.train_samples
        if n not in groups:
            groups[n] = {}
        if r.name not in groups[n] or r.timestamp > groups[n][r.name].timestamp:
            groups[n][r.name] = r
    return dict(sorted(groups.items()))


def print_comparison_table(groups: Dict[int, Dict[str, ExperimentResult]]):
    """Print one comparison table per sample-size group."""
    if not groups:
        print("No results found. Run experiments first.")
        return

    for n_samples, exps in groups.items():
        print("\n" + "="*100)
        print(f"COMPARISON — {n_samples} training samples (latest run per experiment)")
        print("="*100)
        header = f"{'Experiment':<25} {'F1':>8} {'Acc':>7} {'Train(s)':>10} {'Eval(s)':>8} {'Train Loss':>10} {'Eval Loss':>10}"
        print(header)
        print("-"*100)
        for name, r in sorted(exps.items()):
            print(f"{name:<25} {r.f1:>8.3f} {r.accuracy:>7.3f} {r.train_time:>10.1f} {r.eval_time:>8.1f} {r.train_loss:>10.3f} {r.eval_loss:>10.3f}")
        print("="*100)
        best_f1 = max(exps.values(), key=lambda x: x.f1)
        fastest = min(exps.values(), key=lambda x: x.train_time)
        print(f"  Best F1:          {best_f1.name} ({best_f1.f1:.3f})")
        print(f"  Fastest Training: {fastest.name} ({fastest.train_time:.1f}s)")


def plot_comparison(groups: Dict[int, Dict[str, ExperimentResult]], out_dir: str):
    """Generate one comparison PNG per sample-size group."""
    if not HAS_MATPLOTLIB:
        print("\nNote: Install matplotlib for visualization: pip install matplotlib")
        return

    saved = []
    for n_samples, exps in groups.items():
        exp_names = sorted(exps.keys())
        f1s   = [exps[e].f1         for e in exp_names]
        accs  = [exps[e].accuracy   for e in exp_names]
        times = [exps[e].train_time for e in exp_names]

        short_names = [e.replace("exp_", "").replace("_", "\n") for e in exp_names]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Experiment Comparison — {n_samples} training samples", fontsize=13, fontweight='bold')

        axes[0].bar(short_names, f1s, color='steelblue')
        axes[0].set_ylabel('F1 Score')
        axes[0].set_title('F1 Score')
        axes[0].set_ylim(0, 1)

        axes[1].bar(short_names, accs, color='forestgreen')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].set_ylim(0, 1)

        axes[2].bar(short_names, times, color='coral')
        axes[2].set_ylabel('Training Time (s)')
        axes[2].set_title('Training Time')

        plt.tight_layout()
        path = os.path.join(out_dir, f"comparison_{n_samples}samples.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved.append(path)
        print(f"Saved: {path}")

    return saved


def export_csv(results: List[ExperimentResult], csv_path: str):
    """Export all results to CSV."""
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Timestamp', 'Train_Samples', 'F1', 'Accuracy',
                         'Train_Time_s', 'Eval_Time_s', 'Epochs', 'Train_Loss', 'Eval_Loss'])
        for r in results:
            writer.writerow([r.name, r.timestamp, r.train_samples, r.f1, r.accuracy,
                             r.train_time, r.eval_time, r.epochs, r.train_loss, r.eval_loss])
    print(f"CSV exported: {csv_path}")


def main():
    results = load_all_results()

    if not results:
        print("No experiment results found. Run some experiments first!")
        return

    groups = group_by_sample_size(results)

    print(f"\nFound {len(results)} runs across {len(groups)} sample-size group(s): {sorted(groups.keys())}")

    print_comparison_table(groups)
    plot_comparison(groups, out_dir=BENCHMARK_DIR)

    csv_path = os.path.join(BENCHMARK_DIR, "comparison.csv")
    export_csv(results, csv_path)

if __name__ == "__main__":
    main()
