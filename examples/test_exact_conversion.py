import torch
import torch.nn as nn

from tensorfold import TensorFoldLinear


torch.manual_seed(42)

dense = nn.Linear(
    256,
    256
)

tensorfold = TensorFoldLinear.from_linear(
    dense,
    rank=256
)

x = torch.randn(
    32,
    256
)

y_dense = dense(x)
y_tensorfold = tensorfold(x)

mse = torch.mean(
    (y_dense - y_tensorfold) ** 2
)

max_error = torch.max(
    torch.abs(
        y_dense - y_tensorfold
    )
).item()

print("MSE:")
print(mse.item())

print("\nMaximum absolute error:")
print(max_error)

print("\nEquivalent:")
print(
    torch.allclose(
        y_dense,
        y_tensorfold,
        atol=1e-5
    )
)
