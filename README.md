# TensorFold-arm

> Low-rank neural network optimization for efficient inference on Arm-powered CPUs.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## Overview

TensorFold-arm is an experimental framework that automatically compresses dense neural network layers using low-rank matrix factorization.

Instead of executing a single large dense matrix multiplication,

```
Y = XW
```

TensorFold decomposes the weight matrix into two smaller matrices,

```
W ≈ UV
```

and performs

```
Y ≈ (XU)V
```

reducing computation and memory while maintaining model accuracy.

This project was built specifically for the **Arm AI Optimization Challenge** and focuses on improving inference efficiency on Arm CPUs.

> **Note**
>
> TensorFold-arm is an independent research prototype created for the Arm AI Optimization Challenge. It is separate from the TensorFold subsystem being developed inside the MiniPyPy project.

---

# Why TensorFold?

Modern neural networks spend a large fraction of inference time inside fully connected (Linear) layers.

Many of these weight matrices contain significant redundancy.

TensorFold exploits redundancy in neural network weight matrices by automatically converting dense layers into efficient low-rank representations.

This reduces the computational workload and memory footprint of inference while maintaining model accuracy.

The goal is to achieve:

- smaller models
- lower inference latency
- reduced memory usage
- minimal accuracy degradation

without changing the surrounding PyTorch model.

---

# Features

- Automatic analysis of Linear layers
- Rank selection using retained singular-value energy
- Automatic replacement with TensorFoldLinear
- End-to-end model compression
- Accuracy evaluation
- Energy sweep analysis
- CPU benchmarking
- Reproducible experiments

---

# Architecture

Dense inference

```
Input
  │
  ▼
Linear
  │
  ▼
Output
```

TensorFold inference

```
Input
  │
  ▼
Linear A (Input → Rank)
  │
  ▼
Linear B (Rank → Output)
  │
  ▼
Output
```

---

# Results

## MNIST MLP

Architecture

```
784 → 512 → 256 → 10
```

### Accuracy

| Model | Accuracy |
| -------- | ---------- |
| Dense | **97.79%** |
| TensorFold | **97.06%** |

Accuracy change

```
−0.73 percentage points
```

---

### Model Size

| Model | Parameters |
| -------- | -----------: |
| Dense | 535,818 |
| TensorFold | 279,940 |

Parameter reduction

```
47.75%
```

---

### CPU Benchmark (Single Thread)

| Batch | Dense (ms) | TensorFold (ms) | Speedup |
| ------: | -----------: | ----------------: | ---------: |
| 1 | 0.0852 | 0.0820 | 1.039× |
| 16 | 0.3182 | 0.2401 | 1.325× |
| 32 | 0.5031 | 0.3335 | 1.509× |
| 64 | 0.8549 | 0.5702 | 1.499× |
| 256 | 3.0489 | 1.9401 | **1.572×** |

Benchmark configuration

- CPU
- Single thread
- 10 repetitions
- 500 iterations

---

# Installation

```bash
git clone https://github.com/mridul-afk/TensorFold-arm.git

cd TensorFold-arm

pip install -r requirements.txt
```

---

# Train

```bash
python examples/train_mnist.py
```

---

# Analyze

```bash
python examples/analyze_mnist.py
```

---

# Compress

```bash
python examples/compress_mnist.py
```

---

# Evaluate

```bash
python examples/evaluate_compression.py
```

---

# Benchmark

```bash
python examples/benchmark_tensorfold.py
```

---

# Repository Structure

```
tensorfold/
    decomposition.py
    layers.py
    optimizer.py

examples/
    train_mnist.py
    analyze_mnist.py
    compress_mnist.py
    evaluate_compression.py
    energy_sweep.py
    benchmark_tensorfold.py

benchmarks/
    results/

tests/
```

---

# Roadmap

- [ ] Low-rank Linear layer
- [ ] Automatic model compression
- [ ] Energy-based rank selection
- [ ] Accuracy evaluation
- [ ] CPU benchmarking
- [ ] Native Arm benchmarking
- [ ] Automatic latency-aware rank selection
- [ ] Support for larger transformer models
- [ ] Arm-specific kernel optimizations

---

# License

MIT License
