# Bigger Bird: Content-Aware Sparse Transformer Attention

## Overview

Bigger Bird is a research project conducted by the University of Toronto Machine Intelligence Student Team (UTMIST). The primary objective is to reduce the computational complexity of standard Transformer self-attention from $O(n^2)$ to $O(n)$ using content-aware sparse routing.

Standard sparse attention models, such as Big Bird and Longformer, utilize fixed or random attention patterns. Bigger Bird improves upon these by implementing discrete routing mechanisms that adaptively select relevant tokens based on input content, aiming for higher accuracy with significantly fewer softmax comparisons.

## Current Experimental Implementations

The repository contains four distinct experimental approaches to sparse attention, each designed to optimize specific aspects of the attention mechanism:

1.  **DeepSeek-Inspired Top-K Routing (exp_1_deepseek_topk)**: Implements a low-rank projection heuristic to approximate the DeepSeek Lightning Indexer. It projects queries and keys into a low-dimensional space to identify the most relevant tokens before performing high-precision attention on local subsets.
2.  **Lightning Hybrid Attention (exp_2_lightning_hybrid)**: A dual-path approach that combines sharp local attention (standard Softmax) with an efficient linear attention feature map for long-range global context. This maintains local precision while ensuring linear scaling.
3.  **Content-Aware Dynamic Globals (exp_3_dynamic_globals)**: Replaces static global tokens with a learned gating network. This network evaluates the "global importance" of every token in the sequence and dynamically selects the top candidates to facilitate global information broadcasting.
4.  **Permuted Block-Sparse Attention (exp_4_pbs_attn)**: Optimizes for hardware efficiency by calculating affinity at the block level. Tokens are processed in contiguous chunks to maximize GPU memory coalescing and minimize the overhead of random memory access.

## Repository Structure

The project is organized to facilitate fair and consistent benchmarking across different architectures:

*   **shared/**: Contains standardized utility modules used by all experiments.
    *   `dataset.py`: Uniform IMDb dataset loading and preprocessing logic.
    *   `runner.py`: A standardized training and evaluation wrapper using the Hugging Face Trainer API.
*   **exp_*/**: Individual implementation folders for each experimental architecture, containing specific model definitions and execution scripts.
*   **benchmarks/**: Automated output directory for tracking performance metrics (F1 score), training latency, and memory utilization.

## Execution Instructions

To execute the experiments, ensure that the environment requirements (PyTorch, Transformers, Accelerate) are met.

### Running a specific experiment
Navigate to an experiment directory and execute the run script:
```bash
python exp_1_deepseek_topk/run.py
```

### Running all experiments sequentially
To perform a complete sweep of all implementations:
```bash
python exp_1_deepseek_topk/run.py && \
python exp_2_lightning_hybrid/run.py && \
python exp_3_dynamic_globals/run.py && \
python exp_4_pbs_attn/run.py
```

The current micro-benchmark configuration is set to train on 200 samples with a sequence length of 256 for rapid iteration and validation of the routing logic.

## Technical Stack

*   **Frameworks**: Python, PyTorch, Hugging Face Transformers, Accelerate.
*   **Evaluation Metrics**: F1 score (primary), training throughput (steps/sec), and memory footprint.
*   **Baseline**: Experiments are currently validated against the facebook/bart-base architecture as a comparative patch.

## Team

This project is a collaborative effort by the University of Toronto Machine Intelligence Student Team (UTMIST).

*   **Leads**: Derek Chen, Han Fang
*   **Advisors**: Isaac
*   **Developers**: Adrian, Brian, Jerry, Karma, Mohamad, Ryan, Thomas
