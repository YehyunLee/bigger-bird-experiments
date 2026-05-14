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

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "benchmarks")

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

def print_comparison_table(results: List[ExperimentResult]):
    """Print a formatted comparison table."""
    if not results:
        print("No results found. Run experiments first.")
        return
    
    # Group by experiment name, keep latest
    latest_by_exp = {}
    for r in results:
        if r.name not in latest_by_exp or r.timestamp > latest_by_exp[r.name].timestamp:
            latest_by_exp[r.name] = r
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON TABLE (Latest run per experiment)")
    print("="*100)
    
    header = f"{'Experiment':<25} {'F1':>8} {'Acc':>7} {'Train(s)':>10} {'Eval(s)':>8} {'Train Loss':>10} {'Eval Loss':>10}"
    print(header)
    print("-"*100)
    
    for name, r in sorted(latest_by_exp.items()):
        print(f"{name:<25} {r.f1:>8.3f} {r.accuracy:>7.3f} {r.train_time:>10.1f} {r.eval_time:>8.1f} {r.train_loss:>10.3f} {r.eval_loss:>10.3f}")
    
    print("="*100)
    
    # Best by F1
    best_f1 = max(latest_by_exp.values(), key=lambda x: x.f1)
    print(f"\nBest F1: {best_f1.name} ({best_f1.f1:.3f})")
    
    # Fastest training
    fastest = min(latest_by_exp.values(), key=lambda x: x.train_time)
    print(f"Fastest Training: {fastest.name} ({fastest.train_time:.1f}s)")

def plot_comparison(results: List[ExperimentResult], save_path: str = None):
    """Create comparison plots if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        print("\nNote: Install matplotlib for visualization: pip install matplotlib")
        return
    
    # Group by experiment name, keep latest
    latest_by_exp = {}
    for r in results:
        if r.name not in latest_by_exp or r.timestamp > latest_by_exp[r.name].timestamp:
            latest_by_exp[r.name] = r
    
    exps = sorted(latest_by_exp.keys())
    f1s = [latest_by_exp[e].f1 for e in exps]
    accs = [latest_by_exp[e].accuracy for e in exps]
    times = [latest_by_exp[e].train_time for e in exps]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # F1 Score
    axes[0].bar(exps, f1s, color='steelblue')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score by Experiment')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Accuracy
    axes[1].bar(exps, accs, color='forestgreen')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy by Experiment')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Training Time
    axes[2].bar(exps, times, color='coral')
    axes[2].set_ylabel('Training Time (s)')
    axes[2].set_title('Training Time by Experiment')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def export_csv(results: List[ExperimentResult], csv_path: str = "comparison.csv"):
    """Export results to CSV for external analysis."""
    import csv
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Timestamp', 'F1', 'Accuracy', 'Train_Time_s', 
                        'Eval_Time_s', 'Train_Samples', 'Eval_Samples', 'Epochs',
                        'Train_Loss', 'Eval_Loss'])
        
        for r in results:
            writer.writerow([r.name, r.timestamp, r.f1, r.accuracy, r.train_time,
                           r.eval_time, r.train_samples, r.eval_samples, r.epochs,
                           r.train_loss, r.eval_loss])
    
    print(f"Exported to {csv_path}")

def main():
    results = load_all_results()
    
    if not results:
        print("No experiment results found. Run some experiments first!")
        return
    
    print_comparison_table(results)
    
    # Generate plots
    plot_path = os.path.join(BENCHMARK_DIR, "comparison.png")
    plot_comparison(results, save_path=plot_path)
    
    # Export CSV
    csv_path = os.path.join(BENCHMARK_DIR, "comparison.csv")
    export_csv(results, csv_path)
    
    print(f"\nAll results available in: {BENCHMARK_DIR}")

if __name__ == "__main__":
    main()
