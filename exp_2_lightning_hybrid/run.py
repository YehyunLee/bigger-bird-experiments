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
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    base_model.config.classifier_dropout = 0.1
    
    # 2. Patch with Lightning Hybrid
    model = PatchedModel(base_model, block_size=128) 
    
    data_cfg = DataConfig(train_samples=200, eval_samples=50, max_length=256) 
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=256)
    
    train_cfg = TrainConfig(epochs=1, lr=3e-5)
    run_experiment(
        "exp_2_lightning_hybrid", 
        model, 
        tokenizer, 
        ds, 
        train_cfg,
        extra_meta={"block_size": 128}
    )

if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
