# pytorch_intro.py

import torch
import numpy as np

# ---------------------------------------------
# 1. Tensor Initialization
# ---------------------------------------------

# Create a 1D tensor from a Python list


t1 = torch.tensor([1.0, 2.0, 3.0])
print("Tensor t1:", t1)
print(t1.dtype)

t1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
print(t1.dtype)


# Transform from NumPy

t1_np = np.array([1.0, 2.0, 3.0])
t1_torch = torch.from_numpy(t1_np)
print(type(t1_torch))
t1_np = t1_torch.numpy()
print(type(t1_np))

# Create a 2D tensor with specific values
t2 = torch.tensor([[1, 2], [3, 4]])
print("\nTensor t2:\n", t2)
print(t2.dtype)

print(t2.shape)

# Create tensors filled with zeros, ones, or random numbers
z = torch.zeros((2, 3))
o = torch.ones((2, 3))
r = torch.rand((2, 3))
print("\nZeros:\n", z)
print("Ones:\n", o)
print("Random:\n", r)


# ---------------------------------------------
# 2. Basic Operations
# ---------------------------------------------

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("\na + b:", a + b)
print("a * b:", a * b)
print("Dot product:", torch.dot(a, b))
print("Mean of a:", torch.mean(a))

# ---------------------------------------------
# 3. Indexing and Slicing
# ---------------------------------------------

m = torch.tensor([[10, 20, 30], [40, 50, 60]])
print("\nMatrix m:\n", m)
print("First row:", m[0])
print("Second column:", m[:, 1])
print("Element (1,2):", m[1, 2])

# ---------------------------------------------
# 4. Reshaping and Viewing
# ---------------------------------------------

x = torch.arange(12)
print("\nOriginal x:", x)

x_reshaped = x.reshape(3, 4)  # Reshape to 3x4
print("Reshaped x (3x4):\n", x_reshaped)

x_flattened = x_reshaped.reshape(-1)  # Flatten back
print("Flattened x:", x_flattened)

x_transposed = x_reshaped.transpose(0, 1)
print("Transposed x (4x3):\n", x_transposed)

x_reshaped2 = x_reshaped.reshape(3, 2, 2)

x_reshaped3 = x_transposed.reshape(2,2,3)

x_reshaped4 = x_reshaped.reshape(2,2,3)

print(x_reshaped2.reshape(-1))
print(x_reshaped3.reshape(-1))
print(x_reshaped4.reshape(-1))
# ---------------------------------------------
# 5. In-place Operations
# ---------------------------------------------

print("\nOriginal a:", a)
a.add_(10)  # In-place addition
a.mul_(2)   # In-place multiplication
print("Modified a (in-place):", a)

# ---------------------------------------------
# 6. Reductions on a 3D tensor
# ---------------------------------------------

T = torch.randint(0, 10, (2, 3, 4))
print("\n3D Tensor T:\n", T)
print("Sum over last dimension (dim=2):\n", T.sum(dim=2))
print("Max over second dimension (dim=1):\n", T.max(dim=1).values)
print("Mean over first dimension (dim=0):\n", T.float().mean(dim=0))

T.mean(dim=0)

# ---------------------------------------------
# 7. Matrix and Vector Multiplication
# ---------------------------------------------

A = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
v = torch.tensor([1., 0., -1.])
print("\nMatrix A:\n", A)
print("Vector v:", v)
print("Matrix-vector product A @ v:", torch.matmul(A, v))

B = torch.tensor([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]])
print("\nMatrix-matrix product A @ B:\n", torch.matmul(A, B))

# ---------------------------------------------
# 8. Batch-wise Matrix Multiplication
# ---------------------------------------------

batch_A = torch.rand(2, 3, 4)
batch_B = torch.rand(2, 4, 2)
result = torch.bmm(batch_A, batch_B)
print("\nBatch A shape:", batch_A.shape)
print("Batch B shape:", batch_B.shape)
print("Batch-wise product shape:", result.shape)

# ---------------------------------------------
# 9. Bonus: Function Evaluation on Grid
# ---------------------------------------------
x = torch.linspace(-1, 1, steps=100)
y = torch.linspace(-1, 1, steps=100)
X, Y = torch.meshgrid(x, y, indexing='ij')
Z = X**2 + X*Y + Y**2
print("\nMean value of f(x, y) = x^2 + xy + y^2 over grid:", Z.mean().item())