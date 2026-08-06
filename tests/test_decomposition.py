import pytest
import torch

from tensorfold.decomposition import (
    low_rank_svd,
    select_rank,
    analyze_linear,
)


def test_low_rank_svd():
    torch.manual_seed(42)

    weight = torch.randn(32, 16)

    U, S, Vh = low_rank_svd(weight, rank=8)

    assert U.shape == (32, 8)
    assert S.shape == (8,)
    assert Vh.shape == (8, 16)


def test_low_rank_svd_reconstruction():
    torch.manual_seed(42)

    weight = torch.randn(32, 16)

    U, S, Vh = low_rank_svd(weight, rank=16)

    reconstructed = U @ torch.diag(S) @ Vh

    assert torch.allclose(
        reconstructed,
        weight,
        atol=1e-5,
        rtol=1e-5,
    )


def test_low_rank_svd_invalid_rank():
    weight = torch.randn(16, 16)

    with pytest.raises(ValueError):
        low_rank_svd(weight, rank=0)

    with pytest.raises(ValueError):
        low_rank_svd(weight, rank=17)


def test_select_rank():
    torch.manual_seed(42)

    weight = torch.randn(32, 16)

    rank = select_rank(weight, energy=0.95)

    assert isinstance(rank, int)
    assert 1 <= rank <= 16


def test_select_rank_invalid_energy():
    weight = torch.randn(16, 16)

    with pytest.raises(ValueError):
        select_rank(weight, energy=0)

    with pytest.raises(ValueError):
        select_rank(weight, energy=1.1)


def test_analyze_linear():
    torch.manual_seed(42)

    weight = torch.randn(256, 512)

    result = analyze_linear(
        weight,
        energy=0.95,
    )

    assert result["in_features"] == 512
    assert result["out_features"] == 256
    assert result["original_parameters"] == 512 * 256
    assert result["rank"] >= 1
    assert result["compressed_parameters"] > 0
    assert "parameter_reduction" in result
    assert "compression_possible" in result
