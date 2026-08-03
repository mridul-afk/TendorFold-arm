import time

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)


device = torch.device("cpu")



model = MLP().to(device)

model.load_state_dict(
    torch.load(
        "mnist_mlp.pt",
        map_location=device
    )
)

model.eval()




parameters = sum(
    p.numel()
    for p in model.parameters()
)

model_size = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
)



batch_size = 256

x = torch.randn(
    batch_size,
    784,
    device=device
)


# Warmup
with torch.no_grad():
    for _ in range(20):
        model(x)


# Benchmark
iterations = 100

start = time.perf_counter()

with torch.no_grad():
    for _ in range(iterations):
        model(x)

end = time.perf_counter()


latency_ms = (
    (end - start)
    / iterations
    * 1000
)




print("TensorFold Arm Baseline")
print("=======================")

print(f"Device: CPU")

print(
    f"Parameters: {parameters:,}"
)

print(
    f"Model size: "
    f"{model_size / 1024:.2f} KB"
)

print(
    f"Batch size: {batch_size}"
)

print(
    f"Average latency: "
    f"{latency_ms:.4f} ms"
)
