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


device = torch.device("cpu")


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = MLP().to(device)

model.load_state_dict(
    torch.load(
        "mnist_mlp.pt",
        map_location=device,
    )
)

model.eval()


# --------------------------------------------------
# TensorFold compression
# --------------------------------------------------

compressed_model = compress(
    model,
    energy=0.95,
)

compressed_model.eval()


# --------------------------------------------------
# Parameter comparison
# --------------------------------------------------

original_parameters = sum(
    p.numel()
    for p in model.parameters()
)

compressed_parameters = sum(
    p.numel()
    for p in compressed_model.parameters()
)

reduction = (
    1
    - compressed_parameters
    / original_parameters
) * 100


print("TensorFold Compression")
print("======================")

print(
    f"Original parameters: "
    f"{original_parameters:,}"
)

print(
    f"TensorFold parameters: "
    f"{compressed_parameters:,}"
)

print(
    f"Parameter reduction: "
    f"{reduction:.2f}%"
)


# --------------------------------------------------
# Show architecture
# --------------------------------------------------

print("\nCompressed model:")
print(compressed_model)
