import pytest
import torch
import torch.nn as nn

from tensorfold.layers import TensorFoldLinear


def test_tensorfold_linear_shape():
    torch.manual_seed(42)

    layer = TensorFoldLinear(
        in_features=64,
        out_features=32,
        rank=16,
    )

    x = torch.randn(8, 64)

    y = layer(x)

    assert y.shape == (8, 32)


def test_tensorfold_linear_parameter_count():
    layer = TensorFoldLinear(
        in_features=64,
        out_features=32,
        rank=16,
    )

    expected = (
        64 * 16
        + 16 * 32
        + 32
    )

    actual = sum(
        p.numel()
        for p in layer.parameters()
    )

    assert actual == expected


def test_tensorfold_linear_invalid_rank():
    with pytest.raises(ValueError):
        TensorFoldLinear(
            in_features=64,
            out_features=32,
            rank=0,
        )

    with pytest.raises(ValueError):
        TensorFoldLinear(
            in_features=64,
            out_features=32,
            rank=33,
        )


def test_tensorfold_linear_from_linear():
    torch.manual_seed(42)

    dense = nn.Linear(16, 8)

    compressed = TensorFoldLinear.from_linear(
        dense,
        rank=8,
    )

    x = torch.randn(4, 16)

    y_dense = dense(x)
    y_compressed = compressed(x)

    assert y_dense.shape == y_compressed.shape

    assert torch.allclose(
        y_dense,
        y_compressed,
        atol=1e-5,
        rtol=1e-5,
    )


def test_tensorfold_linear_no_bias():
    layer = TensorFoldLinear(
        in_features=16,
        out_features=8,
        rank=4,
        bias=False,
    )

    assert layer.bias is None

    x = torch.randn(2, 16)

    y = layer(x)

    assert y.shape == (2, 8)
