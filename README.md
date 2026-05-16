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

## Architecture Diagrams

### Base Model: BART-base (Standard Transformer)

```
Input Tokens (N=768)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Token Embeddings + Positional Encoding             │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  BART Encoder (12 layers)                           │
│  ┌───────────────────────────────────────────────┐  │
│  │  Each Layer:                                  │  │
│  │                                               │  │
│  │  Input ──► Self-Attention ──► FFN ──► Output │  │
│  │            (O(n²) complexity)                  │  │
│  │                                               │  │
│  │  Attention(Q,K,V) = softmax(QK^T/√d)V         │  │
│  │  Every token attends to ALL other tokens    │  │
│  │  Memory: O(n²), Compute: O(n²)              │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Classification Head                                │
│  └─► Attention Pooling ──► Linear ──► Softmax     │
└─────────────────────────────────────────────────────┘
       │
       ▼
   [Logits]
```

---

### Experiment 1: DeepSeek-Inspired Top-K Routing

**Key Innovation**: Low-rank projection to approximate token importance before full attention.

```
Input ──► Q, K, V projections
              │
              ▼
    ┌─────────────────────┐
    │ Low-Rank Routing    │  ◄── O(n² × d_low) where d_low << d
    │ Q_low, K_low (d=16) │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Rough Scores      │  ──► Top-K Selection (K=128)
    │ Q_low × K_low^T   │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Gather K, V for     │  ◄── Only K tokens selected
    │ selected indices    │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Precise Attention   │  ──► Full-dim softmax over K tokens
    │ softmax(Q × K^T)V   │      O(n × K × d)
    └─────────────────────┘
```

**Complexity**: O(n² × d_low + n × K × d) ≈ O(n) when K << n

---

### Experiment 2: Lightning Hybrid Attention

**Key Innovation**: Dual-path — local precision (softmax) + global efficiency (linear).

```
Input ──► Q, K, V
              │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
┌──────────┐     ┌──────────────┐
│ Local    │     │ Global       │
│ Window   │     │ Linear       │
│ (Softmax)│     │ Attention    │
│          │     │              │
│ Window:  │     │ Q' = elu(Q)+1│
│ |i-j| ≤  │     │ K' = elu(K)+1│
│ block/2  │     │              │
│          │     │ Output =     │
│ O(n×W)   │     │ Q'(K'^T×V)   │
│          │     │ ─────────────│
│          │     │  (K'^T×1)    │
│          │     │ O(n×d²)      │
└────┬─────┘     └──────┬───────┘
     │                  │
     └────────┬─────────┘
              ▼
    ┌─────────────────────┐
    │ Combined Output     │
    │ local_out + 0.5 ×  │  ◄── Tunable mixing coefficient
    │ global_out          │
    └─────────────────────┘
```

**Complexity**: O(n×W + n×d²) = O(n) where W = window size

---

### Experiment 3: Content-Aware Dynamic Globals

**Key Innovation**: Learned gating network selects dynamic global tokens + sliding window.

```
Input ──► hidden_states
              │
              ▼
    ┌─────────────────────┐
    │ Global Gate         │  ◄── Learned linear layer
    │ Linear(embed_dim→1) │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Globalness Scores   │  ──► Per-token importance scores
    │ [B, T]              │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Top-G Selection     │  ──► Select G=16 most "global" tokens
    │ torch.topk(scores,G)│
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────────────────┐
    │ Attention Window Construction │
    │                                 │
    │  ┌─────────┐   ┌───────────┐  │
    │  │ Global  │ + │  Local    │  │  ◄── G globals + W=64 window
    │  │ (G tok) │   │ (W tok)   │  │
    │  │ broadcast│   │ sliding   │  │
    │  │ to all  │   │ per token │  │
    │  └─────────┘   └───────────┘  │
    │                                 │
    │  Total attended tokens: G + W │
    └─────────────────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Sparse Attention      │  ──► Softmax over (G+W) tokens per query
    │ O(n × (G+W) × d)      │      = O(n) when G,W constant
    └─────────────────────┘
```

**Complexity**: O(n × (G + W) × d) = O(n)

---

### Experiment 4: Permuted Block-Sparse Attention (PBS)

**Key Innovation**: Block-level selection for GPU memory coalescing.

```
Input ──► Q, K, V [shape: BH, T, d]
              │
              ▼
    ┌─────────────────────┐
    │ Block Pooling       │  ◄── Average pool to block representations
    │ block_size = 32     │
    │                     │
    │ Q: [BH, T, d]       │
    │   ↓                 │
    │ Q_blocks: [BH,       │
    │   n_blocks, d]       │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Block Affinity      │  ──► Block-to-block similarity scores
    │ Q_blocks × K_blocks^T│     [BH, n_q_blocks, n_k_blocks]
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Top-M Block Select  │  ──► Select M=4 most relevant blocks
    │ torch.topk(scores,M)│
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Token Index Expand  │  ──► Convert block indices → token indices
    │ Block i → tokens    │      [i×B : (i+1)×B]
    │ [i×32 : (i+1)×32]   │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Contiguous Gather   │  ◄── GPU-friendly: coalesced memory reads
    │ torch.gather()      │      M blocks × 32 tokens = 128 tokens
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Attention           │  ──► O(n × M × block_size × d)
    │ O(n × 128 × d)      │      = O(n)
    └─────────────────────┘
```

**Complexity**: O(n × M × B × d) = O(n) where M = num_blocks, B = block_size

**Hardware Benefit**: Contiguous memory access patterns maximize GPU bandwidth.

---

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
