# TensorFold x86 CPU Reference

## Model

MNIST MLP:

784 → 512 → 256 → 10

## TensorFold configuration

Energy: 90%

## Accuracy

Dense: 97.79%
TensorFold: 97.06%

Accuracy change: -0.73 percentage points

## Parameters

Dense: 535,818
TensorFold: 279,940

Parameter reduction: 47.75%

## Single-thread CPU latency

| Batch | Dense (ms) | TensorFold (ms) | Speedup |
| ------: | -----------: | ----------------: | --------: |
| 1 | 0.0852 | 0.0820 | 1.039x |
| 16 | 0.3182 | 0.2401 | 1.325x |
| 32 | 0.5031 | 0.3335 | 1.509x |
| 64 | 0.8549 | 0.5702 | 1.499x |
| 256 | 3.0489 | 1.9401 | 1.572x |

Threads: 1
Repeats: 10
Iterations: 500
