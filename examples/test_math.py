import torch

from tensorfold import TensorFoldLinear


torch.manual_seed(42)


in_features = 128
out_features = 256
rank = 32
batch_size = 16


layer = TensorFoldLinear(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
)


x = torch.randn(
    batch_size,
    in_features
)


# TensorFold implementation
y_tensorfold = layer(x)


# Explicit mathematical implementation
y_manual = (x @ layer.U) @ layer.V

if layer.bias is not None:
    y_manual = y_manual + layer.bias


# Compare
max_error = torch.max(
    torch.abs(
        y_tensorfold - y_manual
    )
).item()


print("TensorFold output shape:")
print(y_tensorfold.shape)

print("\nMaximum absolute error:")
print(max_error)

print("\nMathematical equivalence:")
print(
    torch.allclose(
        y_tensorfold,
        y_manual,
        atol=1e-6
    )
)
