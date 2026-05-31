# Experiment Commands Quick Reference

## List Available Configs
```bash
python run_experiment.py --list
```

## By Compute Size

### SMALL (Quick test / 8GB RAM / ~1GB mem usage)
```bash
# Quick sanity check - runs in ~5 min
python run_experiment.py --exp 0 --size small   # baseline (full attention)
python run_experiment.py --exp 1 --size small
python run_experiment.py --exp 2 --size small
python run_experiment.py --exp 3 --size small
python run_experiment.py --exp 4 --size small
python run_experiment.py --exp 5 --size small
python run_experiment.py --exp 6 --size small   # S2-Attention / HHST (arXiv:2407.17678)
```
**Config:** 500 samples, seq=256, batch=1, 2 epochs

### MEDIUM (Good GPU / 16GB VRAM / ~6GB mem usage)
```bash
# Decent training - runs in ~30 min
python run_experiment.py --exp 1 --size medium
python run_experiment.py --exp 2 --size medium
python run_experiment.py --exp 3 --size medium
python run_experiment.py --exp 4 --size medium
python run_experiment.py --exp 5 --size medium
python run_experiment.py --exp 6 --size medium
```
**Config:** 2000 samples, seq=512, batch=2, accum=4, 3 epochs

### BIG (Big GPU / 24GB+ VRAM / ~12GB mem usage)
```bash
# Full training - runs in ~2 hours
python run_experiment.py --exp 1 --size big
python run_experiment.py --exp 2 --size big
python run_experiment.py --exp 3 --size big
python run_experiment.py --exp 4 --size big
python run_experiment.py --exp 5 --size big
python run_experiment.py --exp 6 --size big
```
**Config:** 6000 samples, seq=768, batch=4, accum=8, 3 epochs

### XL (Full IMDb / Large GPU / ~20GB mem usage)
```bash
# Production run - full dataset
python run_experiment.py --exp 3 --size xl
```
**Config:** 25000 samples, seq=1024, batch=8, accum=16, 5 epochs

## With Custom Overrides

```bash
# Medium but with more epochs
python run_experiment.py --exp 3 --size medium --epochs 5

# Small batch size for low VRAM
python run_experiment.py --exp 3 --size big --batch 2 --accum 16

# Longer sequences with gradient checkpointing (saves memory)
python run_experiment.py --exp 3 --size medium --seq 768 --grad-checkpoint

# Save weights after training so you can re-run eval without retraining
python run_experiment.py --exp 1 --size big --save-weights

# Custom everything
python run_experiment.py --exp 3 --size small \
    --train-samples 1000 \
    --seq 512 \
    --batch 2 \
    --epochs 4 \
    --lr 2e-5
```

## Full Matrix: Experiment × Compute

| Exp | Small | Medium | Big | XL |
|-----|-----------|--------------|---------------|-----------|
| 1 DeepSeek | `python run_experiment.py --exp 1 --size small` | `--exp 1 --size medium` | `--exp 1 --size big` | `--exp 1 --size xl` |
| 2 Lightning | `python run_experiment.py --exp 2 --size small` | `--exp 2 --size medium` | `--exp 2 --size big` | `--exp 2 --size xl` |
| 3 Dynamic Globals | `python run_experiment.py --exp 3 --size small` | `--exp 3 --size medium` | `--exp 3 --size big` | `--exp 3 --size xl` |
| 4 PBS | `python run_experiment.py --exp 4 --size small` | `--exp 4 --size medium` | `--exp 4 --size big` | `--exp 4 --size xl` |
| 5 NSA | `python run_experiment.py --exp 5 --size small` | `--exp 5 --size medium` | `--exp 5 --size big` | `--exp 5 --size xl` |
| 6 S2/HHST | `python run_experiment.py --exp 6 --size small` | `--exp 6 --size medium` | `--exp 6 --size big` | `--exp 6 --size xl` |


## Saving & Reloading Weights

Pass `--save-weights` to persist the model after training:
```bash
python run_experiment.py --exp 1 --size big --save-weights
```
Weights are saved to `benchmarks/<exp_name>/weights_<timestamp>/` and the path is recorded in the matching `eval_<timestamp>.json` under `experiment_metadata.weights_path`.

To reload for eval-only later:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

weights_dir = "benchmarks/exp_1_deepseek_topk/weights_20260516_165432"
model = AutoModelForSequenceClassification.from_pretrained(weights_dir)
tokenizer = AutoTokenizer.from_pretrained(weights_dir)
# Re-apply patches if needed (exp_1-4 only):
# from exp_1_deepseek_topk.model import patch_bart; patch_bart(model)
# For exp 5: from exp_5_nsa.model import PatchedModel  # wrap base_model with NSA patches
```

> **Note:** For patched experiments (exp 1–4) the *base* BART weights are saved (patches are
> re-applied at load time). For the baseline (exp 0) the full model is saved as-is.
>
> **Exp 5:** Same as other patches — use `PatchedModel` or `patch_bart` from `exp_5_nsa.model` after loading weights.
>
> **Exp 6:** S2-Attention / HHST (arXiv:2407.17678) — use `PatchedModel` from `exp_6_s2_hhst.model` after loading weights.

## Visualize Results

```bash
# After running experiments
python viz/training_trajectory_viz.py  # Per-epoch curves
python viz/compare_experiments.py      # Cross-experiment comparison
python viz/scaling_laws_viz.py         # Scaling trends
```

## Memory-Saving Tips

For 8GB RAM machines:
```bash
# Always use small + gradient checkpointing
python run_experiment.py --exp 3 --size small --grad-checkpoint

# If still crashing, reduce sequence length
python run_experiment.py --exp 3 --size small --seq 128
```

For cloud GPU instances:
```bash
# SSH in first
ssh <user>@<server-ip>

# Then run big/xl experiments
python run_experiment.py --exp 3 --size xl
```
