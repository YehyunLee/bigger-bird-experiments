#!/usr/bin/env python3
"""Regenerate embedded benchmark data in dashboard.html from repo JSON files."""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard.html"


def exp_num(name: str) -> int:
    m = re.search(r"exp_(\d+)", name)
    return int(m.group(1)) if m else -1


def round_val(value, places=3):
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, places) if places else value
    return value


def load_csv_rows() -> list[dict]:
    rows = []
    for path in sorted(ROOT.glob("benchmarks/**/eval_*.json")):
        with open(path) as f:
            data = json.load(f)
        meta = data.get("experiment_metadata", {})
        perf = data.get("performance_metrics", {})
        ev = perf.get("eval", {})
        train = perf.get("train", {})
        ds = meta.get("dataset_info", {})
        mc = meta.get("model_config", {})
        env = meta.get("environment", {})
        name = meta.get("name", path.parent.name)
        ts = meta.get("timestamp", path.stem.replace("eval_", ""))
        train_samples = mc.get("train_samples") or ds.get("train_size")
        rows.append(
            {
                "Experiment": name,
                "Timestamp": ts,
                "Train_Samples": train_samples,
                "F1": round_val(ev.get("eval_f1")),
                "Accuracy": round_val(ev.get("eval_accuracy")),
                "Train_Time_s": round_val(perf.get("training_time_seconds"), 1),
                "Eval_Time_s": round_val(ev.get("eval_runtime"), 2),
                "Epochs": meta.get("training_config", {}).get("epochs"),
                "Train_Loss": round_val(train.get("train_loss"), 3),
                "Eval_Loss": round_val(ev.get("eval_loss"), 3),
                "Peak_Memory_MB": round_val(
                    perf.get("peak_memory_mb") or env.get("peak_memory_mb"), 2
                ),
                "Inference_Latency_ms": round_val(perf.get("inference_latency_ms"), 2),
                "Softmax_Comparisons": perf.get("softmax_comparisons"),
            }
        )
    return rows


def load_efficiency() -> list[dict]:
    with open(ROOT / "benchmarks/efficiency_results.json") as f:
        eff_data = json.load(f)

    efficiency = []
    for row in eff_data["results"]:
        efficiency.append(
            {
                "exp_name": row["exp_name"],
                "exp_num": row["exp_num"],
                "seq_length": row["seq_length"],
                "f1": round_val(row.get("f1")),
                "accuracy": round_val(row.get("accuracy")),
                "train_time_s": round_val(row.get("train_time_s"), 2),
                "peak_memory_mb": round_val(row.get("peak_memory_mb"), 2),
                "inference_latency_ms": round_val(row.get("inference_latency_ms"), 2),
                "softmax_comparisons": row.get("softmax_comparisons"),
                "oom": row.get("oom", False),
            }
        )

    existing = {(e["exp_num"], e["seq_length"]) for e in efficiency}
    for path in ROOT.glob("benchmarks/exp_1[123]_*/eval_*.json"):
        with open(path) as f:
            data = json.load(f)
        meta = data["experiment_metadata"]
        perf = data["performance_metrics"]
        ev = perf.get("eval", {})
        name = meta["name"]
        num = exp_num(name)
        seq = meta.get("model_config", {}).get("seq_length")
        if seq not in (1024, 2048, 4096):
            continue
        key = (num, seq)
        if key in existing:
            continue
        efficiency.append(
            {
                "exp_name": name,
                "exp_num": num,
                "seq_length": seq,
                "f1": round_val(ev.get("eval_f1")),
                "accuracy": round_val(ev.get("eval_accuracy")),
                "train_time_s": round_val(perf.get("training_time_seconds"), 2),
                "peak_memory_mb": round_val(
                    perf.get("peak_memory_mb")
                    or meta.get("environment", {}).get("peak_memory_mb"),
                    2,
                ),
                "inference_latency_ms": round_val(perf.get("inference_latency_ms"), 2),
                "softmax_comparisons": perf.get("softmax_comparisons"),
                "oom": False,
            }
        )
        existing.add(key)

    efficiency.sort(key=lambda x: (x["seq_length"], x["exp_num"]))
    return efficiency


def load_complexity() -> dict:
    with open(ROOT / "benchmarks/complexity_results.json") as f:
        cx = json.load(f)

    return {
        "metadata": {
            "timestamp": cx["metadata"]["timestamp"][:19],
            "device": cx["metadata"]["device"],
            "batch_size": "mixed",
            "trials": cx["metadata"]["trials"],
        },
        "results": [
            {
                "exp_num": row["exp_num"],
                "exp_name": row["exp_name"],
                "seq_length": row["seq_length"],
                "time_ms": round(row["time_ms"], 3) if row.get("time_ms") else None,
                "oom": row["oom"],
            }
            for row in cx["results"]
        ],
    }


def js_object(value) -> str:
    return json.dumps(value, separators=(",", ":"))


def js_rows(rows: list[dict]) -> str:
    lines = ["const CSV_ROWS = ["]
    for row in rows:
        parts = []
        for key, val in row.items():
            if val is None:
                parts.append(f"{key}:null")
            elif isinstance(val, str):
                parts.append(f'{key}:"{val}"')
            elif isinstance(val, bool):
                parts.append(f"{key}:{str(val).lower()}")
            else:
                parts.append(f"{key}:{val}")
        lines.append("  {" + ",".join(parts) + "},")
    lines.append("];")
    return "\n".join(lines)


def patch_dashboard(html: str, complexity: dict, efficiency: list[dict], rows: list[dict]) -> str:
    max_exp = max(
        max(exp_num(r["Experiment"]) for r in rows),
        max(e["exp_num"] for e in efficiency),
        max(r["exp_num"] for r in complexity["results"]),
    )

    data_block = "\n".join(
        [
            f"const COMPLEXITY = {js_object(complexity)};",
            "",
            f"const EFFICIENCY = {js_object(efficiency)};",
            "",
            js_rows(rows),
            "",
            f"const MAX_EXP = {max_exp};",
        ]
    )

    html = re.sub(
        r"const COMPLEXITY = \{.*?const CSV_ROWS = \[.*?\];\n*(?:const MAX_EXP = \d+;\n*)?",
        data_block + "\n",
        html,
        count=1,
        flags=re.DOTALL,
    )

    html = html.replace("for(let e=0;e<=10;e++)", "for(let e=0;e<=MAX_EXP;e++)")
    html = re.sub(
        r'<div class="stat"><div class="val">\d+</div><div class="lbl">Experiments \(exp 0–\d+\)</div></div>',
        f'<div class="stat"><div class="val">{max_exp + 1}</div><div class="lbl">Experiments (exp 0–{max_exp})</div></div>',
        html,
        count=1,
    )

    return html


def main() -> None:
    rows = load_csv_rows()
    efficiency = load_efficiency()
    complexity = load_complexity()
    html = DASHBOARD.read_text()
    html = patch_dashboard(html, complexity, efficiency, rows)
    DASHBOARD.write_text(html)
    print(f"Updated {DASHBOARD}")
    print(f"  CSV rows: {len(rows)}")
    print(f"  Efficiency rows: {len(efficiency)}")
    print(f"  Complexity rows: {len(complexity['results'])}")


if __name__ == "__main__":
    main()
