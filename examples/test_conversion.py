import torch
import torch.nn as nn

from tensorfold import TensorFoldLinear


torch.manual_seed(42)


dense = nn.Linear(
    4096,
    4096
)


rank = 512


tensorfold = TensorFoldLinear.from_linear(
    dense,
    rank=rank
)


x = torch.randn(
    32,
    4096
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


dense_parameters = sum(
    p.numel()
    for p in dense.parameters()
)


tensorfold_parameters = sum(
    p.numel()
    for p in tensorfold.parameters()
)


print("Dense parameters:")
print(dense_parameters)


print("\nTensorFold parameters:")
print(tensorfold_parameters)


print("\nParameter reduction:")
print(
    f"{(1 - tensorfold_parameters / dense_parameters) * 100:.2f}%"
)


print("\nMSE:")
print(mse.item())


print("\nMaximum absolute error:")
print(max_error)
