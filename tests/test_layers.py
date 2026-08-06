import torch

from tensorfold.layers import TensorFoldLinear


def test_tensorfold_linear_backward():
    torch.manual_seed(42)

    layer = TensorFoldLinear(
        in_features=16,
        out_features=8,
        rank=4,
    )

    x = torch.randn(
        4,
        16,
        requires_grad=True,
    )

    y = layer(x)

    loss = y.sum()

    loss.backward()

    assert x.grad is not None
    assert layer.U.grad is not None
    assert layer.V.grad is not None

    if layer.bias is not None:
        assert layer.bias.grad is not None
