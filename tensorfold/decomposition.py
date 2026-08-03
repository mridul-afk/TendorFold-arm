import torch


def low_rank_svd(
    weight: torch.Tensor,
    rank: int
):
    """
    Compute a rank-r approximation of a 2D weight matrix.

    W ≈ U_r @ diag(S_r) @ V_r

    Returns:
        U_r, S_r, V_r
    """

    if weight.ndim != 2:
        raise ValueError(
            "low_rank_svd expects a 2D matrix"
        )

    if rank <= 0:
        raise ValueError(
            "rank must be greater than 0"
        )

    max_rank = min(weight.shape)

    if rank > max_rank:
        raise ValueError(
            f"rank {rank} exceeds maximum possible rank "
            f"{max_rank}"
        )

    U, S, Vh = torch.linalg.svd(
        weight,
        full_matrices=False
    )

    return (
        U[:, :rank],
        S[:rank],
        Vh[:rank, :]
    )


def select_rank(
    weight: torch.Tensor,
    energy: float = 0.95
) -> int:
    """
    Select the smallest rank that preserves the
    requested fraction of singular-value energy.
    """

    if weight.ndim != 2:
        raise ValueError(
            "select_rank expects a 2D matrix"
        )

    if not 0.0 < energy <= 1.0:
        raise ValueError(
            "energy must be between 0 and 1"
        )

    _, S, _ = torch.linalg.svd(
        weight,
        full_matrices=False
    )

    energy_values = S ** 2

    cumulative_energy = torch.cumsum(
        energy_values,
        dim=0
    )

    total_energy = energy_values.sum()

    explained_energy = (
        cumulative_energy / total_energy
    )

    rank = torch.searchsorted(
        explained_energy,
        torch.tensor(
            energy,
            device=weight.device
        )
    ).item() + 1

    return int(rank)


def analyze_linear(
    weight: torch.Tensor,
    energy: float = 0.95
) -> dict:
    """
    Analyze whether a Linear layer is worth
    replacing with a low-rank approximation.
    """

    if weight.ndim != 2:
        raise ValueError(
            "analyze_linear expects a 2D weight matrix"
        )

    out_features, in_features = weight.shape

    original_parameters = (
        in_features * out_features
    )

    rank = select_rank(
        weight,
        energy
    )

    compressed_parameters = (
        in_features * rank
        + rank * out_features
    )

    compression_possible = (
        compressed_parameters < original_parameters
    )

    parameter_reduction = (
        1
        - compressed_parameters / original_parameters
    ) * 100

    return {
        "in_features": in_features,
        "out_features": out_features,
        "rank": rank,
        "energy": energy,
        "original_parameters": original_parameters,
        "compressed_parameters": compressed_parameters,
        "parameter_reduction": parameter_reduction,
        "compression_possible": compression_possible,
    }
