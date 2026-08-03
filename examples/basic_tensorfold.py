import torch

from tensorfold import TensorFoldLinear


torch.manual_seed(42)


layer = TensorFoldLinear(
    in_features=4096,
    out_features=4096,
    rank=512,
)


x = torch.randn(
    32,
    4096
)


y = layer(x)


print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(y.shape)

print("\nParameters:")
print(
    sum(
        p.numel()
        for p in layer.parameters()
    )
)
