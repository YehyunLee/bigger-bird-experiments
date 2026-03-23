import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shared.dataset import build_imdb_dataset, DataConfig
from shared.runner import run_experiment, TrainConfig
from model import PatchedModel

def main():
    model_name = "facebook/bart-base"
    
    # 1. Load baseline model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    base_model.config.classifier_dropout = 0.1
    
    # 2. Patch with Dynamic Globals + Window
    # W=64 limits local block context. G=16 lets it broadcast the 16 most important tokens globally.
    model = PatchedModel(base_model, window_size=64, num_globals=16) 
    
    # 3. Build dataset
    data_cfg = DataConfig(train_samples=1000, eval_samples=250, max_length=512) 
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=512)
    
    # 4. Run experiment
    train_cfg = TrainConfig(epochs=1, lr=3e-5)
    run_experiment("exp_3_dynamic_globals", model, tokenizer, ds, train_cfg)

if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
