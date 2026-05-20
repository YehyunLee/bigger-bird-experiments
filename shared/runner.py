import torch
import os
import time
import psutil
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback
from transformers.trainer_utils import EvalPrediction
import json
from datetime import datetime


def _reset_peak_memory():
    """Reset peak memory counters for the active accelerator."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _peak_memory_mb():
    """Return peak allocated memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        # Fallback: RSS of current process
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 ** 2)


def _compute_softmax_comparisons(seq_len, model, extra_meta):
    """Estimate total softmax comparisons across all layers/heads."""
    # BART-base: 12 layers, 12 heads
    n_layers = 12
    n_heads = 12
    base = seq_len * n_layers * n_heads
    meta = extra_meta or {}

    if meta.get("attention") == "full_dense" or model is None:
        # Baseline: every query attends to all n keys
        return base * seq_len

    # Heuristic based on which experiment params are present
    if "top_k" in meta:
        # Exp 1: top_k keys per query
        return base * meta["top_k"]
    elif "block_size" in meta and "num_blocks" in meta:
        # Exp 4: num_blocks * block_size per query
        return base * meta["num_blocks"] * meta["block_size"]
    elif "window_size" in meta and "num_globals" in meta:
        # Exp 3: globals + window per query
        return base * (meta["num_globals"] + meta["window_size"])
    elif "block_size" in meta:
        # Exp 2: Lightning Hybrid — local window only (block_size)
        return base * meta["block_size"]
    return None


def _measure_inference_latency(model, tokenizer, device, seq_len=256, n_trials=10):
    """Measure average forward-pass latency (ms) on synthetic batch."""
    model.eval()
    dummy = tokenizer(
        "This is a test sentence for latency benchmarking. " * 50,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    dummy = {k: v.to(device) for k, v in dummy.items()}

    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            _ = model(**dummy)

    # Timed runs
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(n_trials):
                _ = model(**dummy)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_trials):
                _ = model(**dummy)
        total_ms = (time.perf_counter() - t0) * 1000

    return total_ms / n_trials

@dataclass
class TrainConfig:
    epochs: int = 3
    per_device_train_bs: int = 2
    per_device_eval_bs: int = 2
    grad_accum_steps: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    if isinstance(eval_pred, EvalPrediction):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        preds, labels = eval_pred
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

def device_flags():
    use_cuda = torch.cuda.is_available()
    use_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    fp16 = False
    bf16 = False
    torch_compile = False
    if use_cuda:
        bf16 = torch.cuda.is_bf16_supported()
        fp16 = not bf16
    return fp16, bf16, torch_compile, use_mps

class TrajectoryCallback(TrainerCallback):
    """Callback to capture per-epoch/step metrics for trajectory visualization."""
    def __init__(self):
        self.trajectory = []
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            point = {
                "epoch": state.epoch,
                "step": state.global_step,
                "train_loss": state.log_history[-1].get("loss", None) if state.log_history else None,
                "eval_loss": metrics.get("eval_loss", None),
                "eval_accuracy": metrics.get("eval_accuracy", None),
                "eval_f1": metrics.get("eval_f1", None),
            }
            self.trajectory.append(point)

def run_experiment(exp_name: str, model, tokenizer, ds, cfg: TrainConfig, extra_meta: dict = None, callbacks=None, save_weights: bool = False):
    fp16, bf16, torch_compile, use_mps = device_flags()
    eval_accum = 1 if use_mps else 8
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks", exp_name))
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_bs,
        per_device_eval_batch_size=cfg.per_device_eval_bs,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
        torch_compile=torch_compile,
        optim="adamw_torch",
        eval_accumulation_steps=eval_accum,
    )

    # Setup trajectory tracking
    traj_callback = TrajectoryCallback()
    all_callbacks = [traj_callback] + (callbacks if callbacks else [])
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=64)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=all_callbacks
    )

    _reset_peak_memory()
    print(f"[{exp_name}] Starting training...", flush=True)
    start_time = time.time()
    train_res = trainer.train()
    train_time = time.time() - start_time
    peak_mem_mb = _peak_memory_mb()
    print(f"[{exp_name}] Peak memory: {peak_mem_mb:.1f} MB")

    print(f"[{exp_name}] Evaluating...", flush=True)
    eval_res = trainer.evaluate()

    # Inference latency
    device = next(model.parameters()).device
    seq_len = ds["train"][0]["input_ids"].shape[0] if "input_ids" in ds["train"][0] else 256
    try:
        inf_latency_ms = _measure_inference_latency(model, tokenizer, device, seq_len=seq_len, n_trials=10)
        print(f"[{exp_name}] Inference latency: {inf_latency_ms:.2f} ms/seq")
    except Exception as e:
        inf_latency_ms = None
        print(f"[{exp_name}] Inference latency measurement failed: {e}")

    # Softmax comparison count
    softmax_comparisons = _compute_softmax_comparisons(seq_len, model, extra_meta)
    if softmax_comparisons:
        baseline_comparisons = seq_len * seq_len * 12 * 12  # n² × heads × layers
        reduction_pct = (1 - softmax_comparisons / baseline_comparisons) * 100
        print(f"[{exp_name}] Softmax comparisons: {softmax_comparisons:,} ({reduction_pct:.1f}% vs baseline)")

    # 📝 Prepare Rich Metadata and Structured Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment_metadata": {
            "name": exp_name,
            "timestamp": timestamp,
            "training_config": {
                "epochs": cfg.epochs,
                "batch_size": cfg.per_device_train_bs,
                "accumulation_steps": cfg.grad_accum_steps,
                "learning_rate": cfg.lr,
                "warmup": cfg.warmup_ratio
            },
            "dataset_info": {
                "train_size": len(ds["train"]),
                "eval_size": len(ds["validation"]),
                "max_seq_len": seq_len
            },
            "environment": {
                "use_mps": use_mps,
                "fp16": fp16,
                "peak_memory_mb": peak_mem_mb
            },
            "model_config": extra_meta or {}
        },
        "performance_metrics": {
            "training_time_seconds": train_time,
            "peak_memory_mb": peak_mem_mb,
            "inference_latency_ms": inf_latency_ms,
            "softmax_comparisons": softmax_comparisons,
            "train": train_res.metrics,
            "eval": eval_res,
            "trajectory": traj_callback.trajectory
        }
    }
    
    # Optionally save model weights
    weights_path = None
    if save_weights:
        weights_dir = os.path.join(out_dir, f"weights_{timestamp}")
        os.makedirs(weights_dir, exist_ok=True)
        # Unwrap PatchedModel wrapper to get the underlying HF model if possible
        save_model = getattr(model, "model", model)
        save_model.save_pretrained(weights_dir)
        tokenizer.save_pretrained(weights_dir)
        weights_path = weights_dir
        print(f"[{exp_name}] Weights saved to {weights_dir}")
    
    if weights_path:
        results["experiment_metadata"]["weights_path"] = weights_path

    # Save as timestamped JSON for scaling law analysis
    json_path = os.path.join(out_dir, f"eval_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Also update the legacy results.txt for quick human check
    res_path = os.path.join(out_dir, "results.txt")
    with open(res_path, "w") as f:
        f.write(f"Experiment: {exp_name} | Date: {timestamp}\n")
        f.write(f"Training Time: {train_time:.2f}s\n")
        f.write(f"Accuracy: {eval_res.get('eval_accuracy', 'N/A')}\n")
        f.write(f"F1: {eval_res.get('eval_f1', 'N/A')}\n")

    print(f"[{exp_name}] Results exported to {json_path}")
    return eval_res
