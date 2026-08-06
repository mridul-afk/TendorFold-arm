import torch
import torch.nn as nn

from tensorfold import compress


def test_compress_linear():
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )

    compressed = compress(
        model,
        energy=0.90,
    )

    assert isinstance(compressed, nn.Module)

    x = torch.randn(4, 32)

    y = compressed(x)

    assert y.shape == (4, 8)
