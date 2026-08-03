import time
import statistics

import torch
import torch.nn as nn

from tensorfold import compress


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

device = torch.device("cpu")

torch.set_num_threads(1)

WARMUP = 100
ITERATIONS = 500
REPEATS = 10

BATCH_SIZES = [
    1,
    16,
    32,
    64,
    256,
]


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = MLP().to(device)

model.load_state_dict(
    torch.load(
        "mnist_mlp.pt",
        map_location=device,
    )
)

model.eval()


tensorfold_model = compress(
    model,
    energy=0.90,
)

tensorfold_model.eval()


# --------------------------------------------------
# Statistics
# --------------------------------------------------

def parameter_count(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


dense_parameters = parameter_count(model)

tensorfold_parameters = parameter_count(
    tensorfold_model
)

parameter_reduction = (
    1
    - tensorfold_parameters
    / dense_parameters
) * 100


# --------------------------------------------------
# Benchmark
# --------------------------------------------------

def benchmark(model, batch_size):

    x = torch.randn(
        batch_size,
        784,
        device=device,
    )

    with torch.no_grad():

        for _ in range(WARMUP):
            model(x)

    measurements = []

    for _ in range(REPEATS):

        start = time.perf_counter()

        with torch.no_grad():

            for _ in range(ITERATIONS):
                model(x)

        end = time.perf_counter()

        latency = (
            (end - start)
            / ITERATIONS
            * 1000
        )

        measurements.append(latency)

    return {
        "mean": statistics.mean(measurements),
        "median": statistics.median(measurements),
        "std": statistics.stdev(measurements),
    }


# --------------------------------------------------
# Results
# --------------------------------------------------

print("TensorFold CPU Benchmark")
print("========================")

print(f"Threads: 1")
print(f"Repeats: {REPEATS}")
print(f"Iterations: {ITERATIONS}")

print()

print(
    f"Dense parameters: "
    f"{dense_parameters:,}"
)

print(
    f"TensorFold parameters: "
    f"{tensorfold_parameters:,}"
)

print(
    f"Parameter reduction: "
    f"{parameter_reduction:.2f}%"
)

print()

print(
    f"{'Batch':>8} "
    f"{'Model':<16} "
    f"{'Mean':>12} "
    f"{'Median':>12} "
    f"{'Std':>12} "
    f"{'Speedup':>12}"
)

print("-" * 76)


for batch_size in BATCH_SIZES:

    dense = benchmark(
        model,
        batch_size,
    )

    tensorfold = benchmark(
        tensorfold_model,
        batch_size,
    )

    speedup = (
        dense["mean"]
        / tensorfold["mean"]
    )

    print(
        f"{batch_size:>8} "
        f"{'Dense':<16} "
        f"{dense['mean']:>12.4f} "
        f"{dense['median']:>12.4f} "
        f"{dense['std']:>12.4f} "
        f"{'-':>12}"
    )

    print(
        f"{'':>8} "
        f"{'TensorFold 90%':<16} "
        f"{tensorfold['mean']:>12.4f} "
        f"{tensorfold['median']:>12.4f} "
        f"{tensorfold['std']:>12.4f} "
        f"{speedup:>11.3f}x"
    )

    print()
