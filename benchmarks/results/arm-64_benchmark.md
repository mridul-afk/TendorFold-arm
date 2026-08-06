# TensorFold Native ARM64 Benchmark

## Environment

- Platform: Native ARM64
- Architecture: ARM64
- Device: CPU
- Threads: 1
- Repeats: 10
- Iterations: 500
- Framework: PyTorch
- Model: MNIST MLP
- TensorFold configuration: 90% energy

## Model Compression

| Metric | Dense | TensorFold 90% |
| --- | ---: | ---: |
| Parameters | 535,818 | 279,940 |
| Parameter reduction | — | **47.75%** |

## Inference Benchmark

| Batch | Dense Mean (ms) | TensorFold Mean (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.1914 | 0.1759 | **1.088×** |
| 16 | 0.7866 | 0.5991 | **1.313×** |
| 32 | 1.2195 | 0.8667 | **1.407×** |
| 64 | 2.1408 | 1.4104 | **1.518×** |
| 256 | 7.5964 | 4.5649 | **1.664×** |

## Detailed Results

### Batch 1

- Dense mean: 0.1914 ms
- Dense median: 0.1812 ms
- Dense standard deviation: 0.0252 ms
- TensorFold mean: 0.1759 ms
- TensorFold median: 0.1747 ms
- TensorFold standard deviation: 0.0035 ms
- Speedup: **1.088×**

### Batch 16

- Dense mean: 0.7866 ms
- Dense median: 0.7458 ms
- Dense standard deviation: 0.0983 ms
- TensorFold mean: 0.5991 ms
- TensorFold median: 0.6190 ms
- TensorFold standard deviation: 0.0801 ms
- Speedup: **1.313×**

### Batch 32

- Dense mean: 1.2195 ms
- Dense median: 1.2172 ms
- Dense standard deviation: 0.0500 ms
- TensorFold mean: 0.8667 ms
- TensorFold median: 0.8552 ms
- TensorFold standard deviation: 0.0481 ms
- Speedup: **1.407×**

### Batch 64

- Dense mean: 2.1408 ms
- Dense median: 2.1345 ms
- Dense standard deviation: 0.1028 ms
- TensorFold mean: 1.4104 ms
- TensorFold median: 1.4116 ms
- TensorFold standard deviation: 0.0781 ms
- Speedup: **1.518×**

### Batch 256

- Dense mean: 7.5964 ms
- Dense median: 7.5175 ms
- Dense standard deviation: 0.4200 ms
- TensorFold mean: 4.5649 ms
- TensorFold median: 4.5699 ms
- TensorFold standard deviation: 0.1257 ms
- Speedup: **1.664×**

## Result

On the native ARM64 runner, TensorFold reduced the model from **535,818 to 279,940 parameters**, corresponding to a **47.75% parameter reduction**.

Inference speedup increased with batch size, reaching a maximum measured speedup of **1.664× at batch size 256**.

The benchmark was performed using **1 CPU thread, 10 repeats, and 500 iterations per measurement**.
