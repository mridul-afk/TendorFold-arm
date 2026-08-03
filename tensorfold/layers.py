import torch
import torch.nn as nn


class TensorFoldLinear(nn.Module):
    """
    Factorized Linear layer.

    Instead of storing a dense weight matrix W:

        Y = XW + b

    TensorFold stores:

        U: [in_features, rank]
        V: [rank, out_features]

    and computes:

        Y = (X @ U) @ V + b
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError(
                "rank must be greater than 0"
            )

        if rank > min(in_features, out_features):
            raise ValueError(
                "rank cannot exceed "
                "min(in_features, out_features)"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.U = nn.Parameter(
            torch.empty(
                in_features,
                rank
            )
        )

        self.V = nn.Parameter(
            torch.empty(
                rank,
                out_features
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features)
            )
        else:
            self.register_parameter(
                "bias",
                None
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.U,
            a=5 ** 0.5
        )

        nn.init.kaiming_uniform_(
            self.V,
            a=5 ** 0.5
        )

        if self.bias is not None:
            bound = 1 / self.in_features ** 0.5

            nn.init.uniform_(
                self.bias,
                -bound,
                bound
            )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        y = x @ self.U
        y = y @ self.V

        if self.bias is not None:
            y = y + self.bias

        return y

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        rank: int,
    ):
        """
        Convert a PyTorch nn.Linear layer into
        a TensorFoldLinear layer using truncated SVD.
        """

        if not isinstance(layer, nn.Linear):
            raise TypeError(
                "from_linear expects an nn.Linear layer"
            )

        if rank <= 0:
            raise ValueError(
                "rank must be greater than 0"
            )

        if rank > min(
            layer.in_features,
            layer.out_features
        ):
            raise ValueError(
                "rank cannot exceed "
                "min(in_features, out_features)"
            )

        # PyTorch weight:
        #
        # [out_features, in_features]
        #
        # TensorFold wants:
        #
        # [in_features, out_features]

        weight = layer.weight.data.T

        # Wᵀ ≈ U @ diag(S) @ Vᵀ

        U, S, Vh = torch.linalg.svd(
            weight,
            full_matrices=False
        )

        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        # Fold singular values into U.
        #
        # U:  [in_features, rank]
        # S:  [rank]

        U = U * S.unsqueeze(0)

        # Vh:
        #
        # [rank, out_features]

        V = Vh

        # Create TensorFold layer

        new_layer = cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            rank=rank,
            bias=layer.bias is not None,
        )

        # Copy factors

        new_layer.U.data.copy_(U)
        new_layer.V.data.copy_(V)

        # Copy bias

        if layer.bias is not None:
            new_layer.bias.data.copy_(
                layer.bias.data
            )

        return new_layer
