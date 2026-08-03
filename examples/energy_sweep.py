import torch
import torch.nn as nn

from tensorfold import compress

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------------------------------
# Model
# --------------------------------------------------

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

energy_levels = [
    0.90,
    0.95,
    0.97,
    0.98,
    0.99,
]


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
# Dataset
# --------------------------------------------------

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model):

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            images = images.view(
                images.size(0),
                -1,
            )

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100.0 * correct / total


# --------------------------------------------------
# Baseline
# --------------------------------------------------

dense_accuracy = evaluate(model)

dense_parameters = sum(
    p.numel()
    for p in model.parameters()
)


# --------------------------------------------------
# Sweep
# --------------------------------------------------

print("TensorFold Energy Sweep")
print("=======================")

print(
    f"Dense accuracy: "
    f"{dense_accuracy:.2f}%"
)

print(
    f"Dense parameters: "
    f"{dense_parameters:,}"
)

print()

print(
    f"{'Energy':>8} "
    f"{'Accuracy':>12} "
    f"{'Δ Accuracy':>12} "
    f"{'Parameters':>14} "
    f"{'Reduction':>12}"
)

print("-" * 64)


for energy in energy_levels:

    compressed_model = compress(
        model,
        energy=energy,
    )

    compressed_model.eval()

    accuracy = evaluate(
        compressed_model
    )

    parameters = sum(
        p.numel()
        for p in compressed_model.parameters()
    )

    reduction = (
        1
        - parameters / dense_parameters
    ) * 100

    accuracy_change = (
        accuracy - dense_accuracy
    )

    print(
        f"{energy * 100:7.1f}% "
        f"{accuracy:11.2f}% "
        f"{accuracy_change:+11.2f}% "
        f"{parameters:13,} "
        f"{reduction:11.2f}%"
    )
