# Experiment Commands Quick Reference

## List Available Configs
```bash
python run_experiment.py --list
```

## By Compute Size

### SMALL (Quick test / 8GB RAM / ~1GB mem usage)
```bash
# Quick sanity check - runs in ~5 min
python run_experiment.py --exp 1 --size small
python run_experiment.py --exp 2 --size small
python run_experiment.py --exp 3 --size small
python run_experiment.py --exp 4 --size small
```
**Config:** 500 samples, seq=256, batch=1, 2 epochs

### MEDIUM (Good GPU / 16GB VRAM / ~6GB mem usage)
```bash
# Decent training - runs in ~30 min
python run_experiment.py --exp 1 --size medium
python run_experiment.py --exp 2 --size medium
python run_experiment.py --exp 3 --size medium
python run_experiment.py --exp 4 --size medium
```
**Config:** 2000 samples, seq=512, batch=2, accum=4, 3 epochs

### BIG (Big GPU / 24GB+ VRAM / ~12GB mem usage)
```bash
# Full training - runs in ~2 hours
python run_experiment.py --exp 1 --size big
python run_experiment.py --exp 2 --size big
python run_experiment.py --exp 3 --size big
python run_experiment.py --exp 4 --size big
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
