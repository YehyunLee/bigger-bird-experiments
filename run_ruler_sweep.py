#!/usr/bin/env python3
"""RULER-style context × depth × experiment sweep.

Runs all 13 sparse-attention methods across sequence lengths and needle depths,
measuring retrieval accuracy (10-way passkey), memory, latency, and OOM survival.

Usage:
  python run_ruler_sweep.py --tasks niah --exps 0,1,7 --seqs 2048,4096 --depths 0.1,0.5,0.9 --size ruler-smoke
  python run_ruler_sweep.py --size ruler-report --skip-existing
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_experiment import EXPERIMENT_CONFIGS
from run_ruler_experiment import RULER_COMPUTE

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(ROOT, "benchmarks")

ALL_EXPS = sorted(EXPERIMENT_CONFIGS.keys())
DEFAULT_EXPS = ",".join(str(x) for x in ALL_EXPS)

DEFAULT_SEQS = {"niah": [2048, 4096, 8192], "mq_niah": [2048, 4096, 8192]}
DEFAULT_DEPTHS = [0.1, 0.5, 0.9]


def latest_eval(task, exp_name):
    d = os.path.join(BENCH, f"ruler_{task}_{exp_name}")
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.startswith("eval_") and f.endswith(".json"))
    if not files:
        return None
    with open(os.path.join(d, files[-1])) as f:
        return json.load(f)


def has_existing_run(task, exp_name, seq, depth, train_samples):
    d = os.path.join(BENCH, f"ruler_{task}_{exp_name}")
    if not os.path.isdir(d):
        return False
    for fname in os.listdir(d):
        if not (fname.startswith("eval_") and fname.endswith(".json")):
            continue
        with open(os.path.join(d, fname)) as f:
            data = json.load(f)
        meta = data.get("experiment_metadata", {})
        mc = meta.get("model_config", {})
        ds = meta.get("dataset_info", {})
        if (meta.get("seq_length") == seq
                and abs(mc.get("needle_depth", -1) - depth) < 1e-6
                and ds.get("train_size") == train_samples):
            return True
    return False


def _load_matching_eval(task, exp_name, seq, depth, train_samples):
    d = os.path.join(BENCH, f"ruler_{task}_{exp_name}")
    if not os.path.isdir(d):
        return None
    best = None
    for fname in os.listdir(d):
        if not (fname.startswith("eval_") and fname.endswith(".json")):
            continue
        with open(os.path.join(d, fname)) as f:
            data = json.load(f)
        meta = data.get("experiment_metadata", {})
        mc = meta.get("model_config", {})
        ds = meta.get("dataset_info", {})
        if (meta.get("seq_length") == seq
                and abs(mc.get("needle_depth", -1) - depth) < 1e-6
                and ds.get("train_size") == train_samples):
            if best is None or meta.get("timestamp", "") > best[0]:
                best = (meta.get("timestamp", ""), data)
    return best[1] if best else None


def run_single(task, exp, seq, depth, size, cpu, skip_existing=False):
    exp_name = EXPERIMENT_CONFIGS[exp][0]
    train_samples = RULER_COMPUTE[size]["train_samples"]

    if skip_existing and has_existing_run(task, exp_name, seq, depth, train_samples):
        print(f"\n{'='*70}\nRULER sweep: {task} | {exp_name} | seq={seq} depth={depth} — SKIP\n{'='*70}")
        data = _load_matching_eval(task, exp_name, seq, depth, train_samples)
        if data is None:
            return {"task": task, "exp": exp, "exp_name": exp_name, "seq": seq,
                    "depth": depth, "status": "skipped"}
        perf = data.get("performance_metrics", {})
        ev = perf.get("eval", {})
        acc = ev.get("eval_accuracy")
        return {
            "task": task, "exp": exp, "exp_name": exp_name, "seq": seq, "depth": depth,
            "accuracy": acc, "f1": ev.get("eval_f1"),
            "train_time_s": perf.get("training_time_seconds"),
            "peak_mem_mb": perf.get("peak_memory_mb"),
            "inference_ms": perf.get("inference_latency_ms"),
            "softmax_comparisons": perf.get("softmax_comparisons"),
            "oom": False, "status": "skipped",
        }

    cmd = [
        sys.executable, "run_ruler_experiment.py",
        "--task", task, "--exp", str(exp), "--seq", str(seq),
        "--depth", str(depth), "--size", size,
    ]
    if cpu:
        cmd.append("--cpu")

    print(f"\n{'='*70}\nRULER sweep: {task} | {exp_name} | seq={seq} depth={depth}\n{'='*70}")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    if res.returncode != 0:
        blob = (res.stderr or "") + (res.stdout or "")
        is_oom = ("out of memory" in blob.lower()) or ("CUDA out of memory" in blob)
        return {
            "task": task, "exp": exp, "exp_name": exp_name, "seq": seq, "depth": depth,
            "oom": is_oom, "status": "oom" if is_oom else "fail",
        }

    data = _load_matching_eval(task, exp_name, seq, depth, train_samples)
    if data is None:
        return {"task": task, "exp": exp, "exp_name": exp_name, "seq": seq, "depth": depth,
                "status": "no_output"}

    perf = data.get("performance_metrics", {})
    ev = perf.get("eval", {})
    acc = ev.get("eval_accuracy")
    collapsed = acc is not None and acc < 0.15  # chance for 10-way ≈ 0.10
    return {
        "task": task, "exp": exp, "exp_name": exp_name, "seq": seq, "depth": depth,
        "accuracy": acc, "f1": ev.get("eval_f1"),
        "train_time_s": perf.get("training_time_seconds"),
        "peak_mem_mb": perf.get("peak_memory_mb"),
        "inference_ms": perf.get("inference_latency_ms"),
        "softmax_comparisons": perf.get("softmax_comparisons"),
        "oom": False,
        "status": "collapse" if collapsed else "ok",
    }


def main():
    parser = argparse.ArgumentParser(description="RULER depth × context × experiment sweep")
    parser.add_argument("--tasks", default="niah,mq_niah", help="Comma-separated tasks")
    parser.add_argument("--exps", default=DEFAULT_EXPS, help="Experiment numbers (default: all 13)")
    parser.add_argument("--seqs", default="", help="Override seq lengths for all tasks")
    parser.add_argument("--depths", default="", help="Override needle depths (e.g. 0.1,0.5,0.9)")
    parser.add_argument("--size", default="ruler-report", help="Compute preset")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    exps = [int(x) for x in args.exps.split(",") if x.strip()]
    seq_override = [int(x) for x in args.seqs.split(",")] if args.seqs else None
    depths = [float(x) for x in args.depths.split(",")] if args.depths else DEFAULT_DEPTHS

    all_results = []
    for task in tasks:
        seqs = seq_override or DEFAULT_SEQS.get(task, [4096])
        for seq in seqs:
            for depth in depths:
                for exp in exps:
                    all_results.append(
                        run_single(task, exp, seq, depth, args.size, args.cpu, args.skip_existing)
                    )

    print("\n" + "=" * 120)
    print("RULER DEPTH × CONTEXT SWEEP SUMMARY")
    print("=" * 120)
    hdr = (f"{'Task':<10} {'Exp':<22} {'Seq':>6} {'Depth':>6} {'Acc':>7} {'F1':>7} "
           f"{'Time(s)':>9} {'Mem(MB)':>9} {'Status':>9}")
    print(hdr)
    print("-" * 120)
    for r in all_results:
        acc = f"{r['accuracy']:.3f}" if r.get("accuracy") is not None else "N/A"
        f1 = f"{r['f1']:.3f}" if r.get("f1") is not None else "N/A"
        t = f"{r['train_time_s']:.1f}" if r.get("train_time_s") else "N/A"
        mem = f"{r['peak_mem_mb']:.0f}" if r.get("peak_mem_mb") else "N/A"
        print(f"{r['task']:<10} {r['exp_name']:<22} {r['seq']:>6} {r['depth']:>6.2f} "
              f"{acc:>7} {f1:>7} {t:>9} {mem:>9} {r.get('status','?'):>9}")
    print("=" * 120)

    os.makedirs(BENCH, exist_ok=True)
    payload = {
        "config": {
            "tasks": tasks, "exps": exps, "seqs": seq_override, "depths": depths,
            "size": args.size, "timestamp": datetime.now().isoformat(),
        },
        "results": all_results,
    }
    ts_path = os.path.join(BENCH, f"ruler_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    latest_path = os.path.join(BENCH, "ruler_sweep_results.json")
    for p in (ts_path, latest_path):
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
    print(f"\nSaved sweep results to: {ts_path}\nAlso updated: {latest_path}")

    try:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_dashboard.py")], cwd=ROOT, check=False)
    except Exception as e:
        print(f"(dashboard rebuild skipped: {e})")


if __name__ == "__main__":
    main()
