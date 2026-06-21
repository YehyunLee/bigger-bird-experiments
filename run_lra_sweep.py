#!/usr/bin/env python3
"""LRA context-window sweep: tasks x sequence lengths x experiments.

The core long-range experiment. Unlike the padded-IMDb sweep, each point measures
accuracy on a task with genuine long-range dependencies, so accuracy degradation under
sparsity is meaningful. Captures accuracy/F1, peak memory, latency, softmax comparisons,
and flags collapse (F1 ~ 0) and OOM -- matching the dashboard's survival/reliability view.

Usage:
  python run_lra_sweep.py --tasks listops,text --exps 0,1,7 --size lra-smoke
  python run_lra_sweep.py --tasks listops --exps 0,1,2,3,4,5,6,7,8,9,10 --seqs 512,1024,2048
  python run_lra_sweep.py --tasks retrieval --exps 0,1,7 --data-dir lra_data/retrieval
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_experiment import EXPERIMENT_CONFIGS
from run_lra_experiment import LRA_COMPUTE

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(ROOT, "benchmarks")

ALL_EXPS = sorted(EXPERIMENT_CONFIGS.keys())
DEFAULT_EXPS = ",".join(str(x) for x in ALL_EXPS)

# Default context windows per task (genuine long-range).
DEFAULT_SEQS = {
    "listops": [512, 1024, 2048],
    "text": [1024, 2048, 4096],
    "retrieval": [1024, 2048, 4096],
}


def latest_eval(task, exp_name):
    d = os.path.join(BENCH, f"lra_{task}_{exp_name}")
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.startswith("eval_") and f.endswith(".json"))
    if not files:
        return None
    with open(os.path.join(d, files[-1])) as f:
        return json.load(f)


def has_existing_run(task, exp_name, seq, train_samples):
    """True if an eval artifact already exists for this (task, exp, seq, budget)."""
    d = os.path.join(BENCH, f"lra_{task}_{exp_name}")
    if not os.path.isdir(d):
        return False
    for fname in os.listdir(d):
        if not (fname.startswith("eval_") and fname.endswith(".json")):
            continue
        with open(os.path.join(d, fname)) as f:
            data = json.load(f)
        meta = data.get("experiment_metadata", {})
        if meta.get("seq_length") == seq and meta.get("dataset_info", {}).get("train_size") == train_samples:
            return True
    return False


def run_single(task, exp, seq, size, data_dir, cpu, skip_existing=False):
    exp_name = EXPERIMENT_CONFIGS[exp][0]
    train_samples = LRA_COMPUTE[size]["train_samples"]
    if skip_existing and has_existing_run(task, exp_name, seq, train_samples):
        print(f"\n{'='*70}\nLRA sweep: {task} | {exp_name} | seq={seq} — SKIP (existing result)\n{'='*70}")
        data = latest_eval(task, exp_name)
        if data is None:
            return {"task": task, "exp": exp, "exp_name": exp_name, "seq": seq, "status": "skipped"}
        perf = data.get("performance_metrics", {})
        ev = perf.get("eval", {})
        meta = data.get("experiment_metadata", {})
        if meta.get("seq_length") != seq:
            # latest_eval may be a different seq; scan for the matching file
            d = os.path.join(BENCH, f"lra_{task}_{exp_name}")
            data = None
            for fname in os.listdir(d):
                if not (fname.startswith("eval_") and fname.endswith(".json")):
                    continue
                with open(os.path.join(d, fname)) as f:
                    cand = json.load(f)
                m = cand.get("experiment_metadata", {})
                if m.get("seq_length") == seq and m.get("dataset_info", {}).get("train_size") == train_samples:
                    data = cand
                    break
            if data is None:
                return {"task": task, "exp": exp, "exp_name": exp_name, "seq": seq, "status": "skipped"}
            perf = data.get("performance_metrics", {})
            ev = perf.get("eval", {})
        f1 = ev.get("eval_f1")
        return {
            "task": task, "exp": exp, "exp_name": exp_name, "seq": seq,
            "accuracy": ev.get("eval_accuracy"), "f1": f1,
            "train_time_s": perf.get("training_time_seconds"),
            "peak_mem_mb": perf.get("peak_memory_mb"),
            "inference_ms": perf.get("inference_latency_ms"),
            "softmax_comparisons": perf.get("softmax_comparisons"),
            "oom": False, "status": "skipped",
        }

    cmd = [
        sys.executable, "run_lra_experiment.py",
        "--task", task, "--exp", str(exp), "--seq", str(seq), "--size", size,
    ]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    if cpu:
        cmd.append("--cpu")

    print(f"\n{'='*70}\nLRA sweep: {task} | {exp_name} | seq={seq}\n{'='*70}")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    if res.returncode != 0:
        # Distinguish a true CUDA/CPU out-of-memory from other crashes.
        blob = (res.stderr or "") + (res.stdout or "")
        is_oom = ("out of memory" in blob.lower()) or ("CUDA out of memory" in blob)
        status = "oom" if is_oom else "fail"
        return {
            "task": task, "exp": exp, "exp_name": exp_name, "seq": seq,
            "oom": is_oom, "status": status,
        }

    data = latest_eval(task, exp_name)
    if data is None:
        return {"task": task, "exp": exp, "exp_name": exp_name, "seq": seq, "oom": False, "status": "no_output"}

    perf = data.get("performance_metrics", {})
    ev = perf.get("eval", {})
    f1 = ev.get("eval_f1")
    collapsed = f1 is not None and f1 < 0.05
    return {
        "task": task,
        "exp": exp,
        "exp_name": exp_name,
        "seq": seq,
        "accuracy": ev.get("eval_accuracy"),
        "f1": f1,
        "train_time_s": perf.get("training_time_seconds"),
        "peak_mem_mb": perf.get("peak_memory_mb"),
        "inference_ms": perf.get("inference_latency_ms"),
        "softmax_comparisons": perf.get("softmax_comparisons"),
        "oom": False,
        "status": "collapse" if collapsed else "ok",
    }


def main():
    parser = argparse.ArgumentParser(description="LRA context-window sweep")
    parser.add_argument("--tasks", default="listops,text", help="Comma-separated LRA tasks")
    parser.add_argument("--exps", default=DEFAULT_EXPS,
                        help=f"Comma-separated experiment numbers (default: all {len(ALL_EXPS)})")
    parser.add_argument("--seqs", default="", help="Override seq lengths (applies to all tasks)")
    parser.add_argument("--size", default="lra-report", help="Compute preset (default: lra-report)")
    parser.add_argument("--data-dir", default=None, help="Data dir for retrieval (AAN)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse existing eval artifacts for the same task/exp/seq/budget")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    exps = [int(x) for x in args.exps.split(",") if x.strip()]
    seq_override = [int(x) for x in args.seqs.split(",")] if args.seqs else None

    all_results = []
    for task in tasks:
        seqs = seq_override or DEFAULT_SEQS.get(task, [1024, 2048])
        for seq in seqs:
            for exp in exps:
                all_results.append(
                    run_single(task, exp, seq, args.size, args.data_dir, args.cpu, args.skip_existing)
                )

    # Summary table
    print("\n" + "=" * 110)
    print("LRA CONTEXT-WINDOW SWEEP SUMMARY")
    print("=" * 110)
    hdr = f"{'Task':<10} {'Exp':<22} {'Seq':>6} {'Acc':>7} {'F1':>7} {'Time(s)':>9} {'Mem(MB)':>9} {'Inf(ms)':>8} {'Status':>9}"
    print(hdr)
    print("-" * 110)
    for r in all_results:
        acc = f"{r['accuracy']:.3f}" if r.get("accuracy") is not None else "N/A"
        f1 = f"{r['f1']:.3f}" if r.get("f1") is not None else "N/A"
        t = f"{r['train_time_s']:.1f}" if r.get("train_time_s") else "N/A"
        mem = f"{r['peak_mem_mb']:.0f}" if r.get("peak_mem_mb") else "N/A"
        inf = f"{r['inference_ms']:.1f}" if r.get("inference_ms") else "N/A"
        print(f"{r['task']:<10} {r['exp_name']:<22} {r['seq']:>6} {acc:>7} {f1:>7} {t:>9} {mem:>9} {inf:>8} {r.get('status','?'):>9}")
    print("=" * 110)

    os.makedirs(BENCH, exist_ok=True)
    payload = {
        "config": {
            "tasks": tasks,
            "exps": exps,
            "seqs": seq_override,
            "size": args.size,
            "timestamp": datetime.now().isoformat(),
        },
        "results": all_results,
    }
    ts_path = os.path.join(BENCH, f"lra_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    latest_path = os.path.join(BENCH, "lra_sweep_results.json")
    for p in (ts_path, latest_path):
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
    print(f"\nSaved sweep results to: {ts_path}\nAlso updated: {latest_path}")

    try:
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_dashboard.py")], cwd=ROOT, check=False)
    except Exception as e:
        print(f"(dashboard rebuild skipped: {e})")


if __name__ == "__main__":
    main()
