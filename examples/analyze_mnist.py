import torch
import torch.nn as nn

from tensorfold import analyze_linear


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


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

device = torch.device("cpu")

model = MLP().to(device)

model.load_state_dict(
    torch.load(
        "mnist_mlp.pt",
        map_location=device
    )
)

model.eval()


# --------------------------------------------------
# Analyze Linear layers
# --------------------------------------------------

energy = 0.95

print("TensorFold Model Analysis")
print("=========================")

total_original = 0
total_compressed = 0

for name, layer in model.named_modules():

    if not isinstance(layer, nn.Linear):
        continue

    result = analyze_linear(
        layer.weight,
        energy=energy
    )

    # Include bias in the actual model parameter count
    original_params = sum(
        p.numel()
        for p in layer.parameters()
    )

    compressed_params = (
        layer.in_features * result["rank"]
        + result["rank"] * layer.out_features
    )

    if layer.bias is not None:
        compressed_params += layer.out_features

    reduction = (
        1
        - compressed_params / original_params
    ) * 100

    total_original += original_params
    total_compressed += compressed_params

    print(f"\nLayer: {name}")
    print(
        f"  Shape: "
        f"{layer.in_features} → "
        f"{layer.out_features}"
    )
    print(
        f"  Target energy: "
        f"{energy * 100:.1f}%"
    )
    print(
        f"  Selected rank: "
        f"{result['rank']}"
    )
    print(
        f"  Original parameters: "
        f"{original_params:,}"
    )
    print(
        f"  TensorFold parameters: "
        f"{compressed_params:,}"
    )
    print(
        f"  Parameter reduction: "
        f"{reduction:.2f}%"
    )
    print(
        f"  Decision: "
        f"{'COMPRESS' if compressed_params < original_params else 'SKIP'}"
    )


# --------------------------------------------------
# Overall Linear-layer statistics
# --------------------------------------------------

overall_reduction = (
    1
    - total_compressed / total_original
) * 100

print("\n=========================")
print("Linear Layers Total")
print("=========================")

print(
    f"Original parameters: "
    f"{total_original:,}"
)

print(
    f"TensorFold parameters: "
    f"{total_compressed:,}"
)

print(
    f"Potential reduction: "
    f"{overall_reduction:.2f}%"
)
