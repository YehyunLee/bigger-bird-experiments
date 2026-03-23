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
    
    # 4. Patch with PBS-Attn Block sizing
    model = PatchedModel(base_model, block_size=64, num_blocks=2) 
    
    data_cfg = DataConfig(train_samples=1000, eval_samples=250, max_length=512) 
    ds = build_imdb_dataset(tokenizer, data_cfg, fixed_length=512)
    
    train_cfg = TrainConfig(epochs=1, lr=3e-5)
    run_experiment("exp_4_pbs_attn", model, tokenizer, ds, train_cfg)

if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
