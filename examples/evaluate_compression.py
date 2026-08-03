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


# --------------------------------------------------
# Create TensorFold model
# --------------------------------------------------

compressed_model = compress(
    model,
    energy=0.95,
)

compressed_model.eval()


# --------------------------------------------------
# MNIST test dataset
# --------------------------------------------------

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
# Evaluation function
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
# Evaluate
# --------------------------------------------------

dense_accuracy = evaluate(model)

tensorfold_accuracy = evaluate(
    compressed_model
)


# --------------------------------------------------
# Parameter comparison
# --------------------------------------------------

dense_parameters = sum(
    p.numel()
    for p in model.parameters()
)

tensorfold_parameters = sum(
    p.numel()
    for p in compressed_model.parameters()
)

reduction = (
    1
    - tensorfold_parameters
    / dense_parameters
) * 100


# --------------------------------------------------
# Results
# --------------------------------------------------

print("TensorFold Evaluation")
print("=====================")

print(
    f"Dense accuracy: "
    f"{dense_accuracy:.2f}%"
)

print(
    f"TensorFold accuracy: "
    f"{tensorfold_accuracy:.2f}%"
)

print(
    f"Accuracy change: "
    f"{tensorfold_accuracy - dense_accuracy:+.2f}%"
)

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
    f"{reduction:.2f}%"
)
