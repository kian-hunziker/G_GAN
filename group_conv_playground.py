import torch
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4
import numpy as np
import matplotlib.pyplot as plt

# Create a 28x28 array filled with -1s
array = -1 * np.ones((28, 28))

# Define the dimensions of the central 7x7 square
square_size = 17

# Calculate the starting and ending indices for the square
start_row = (array.shape[0] - square_size) // 2
end_row = start_row + square_size
start_col = (array.shape[1] - square_size) // 2
end_col = start_col + square_size

# Fill the edges of the central square with ones
array[start_row, start_col:end_col] = 1
array[end_row - 1, start_col:end_col] = 1
array[start_row:end_row, start_col] = 1
array[start_row:end_row, end_col - 1] = 1

# Print the resulting array
plt.imshow(array, cmap='gray')
plt.title('input')
plt.show()

array_tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).float()
print(f'array_tensor.shape: {array_tensor.shape}')

test_conv = P4ConvZ2(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
rand_kernel = test_conv.weight[0][0][0].detach().numpy()

line_kernel = np.zeros((3, 3))
line_kernel[1] = np.ones(3)

# plot kernels
fig, axes = plt.subplots(1, 2)
axes[0].imshow(rand_kernel, cmap='gray')
axes[1].imshow(line_kernel, cmap='gray')
plt.tight_layout()
plt.title('rand kernel left, line kernel right')
plt.show()

# set conv kernel to line
line_kernel_torch = torch.from_numpy(line_kernel).float()
line_kernel_torch.requires_grad = True
with torch.no_grad():
    test_conv.weight[0][0][0] = line_kernel_torch

out = test_conv(array_tensor)
print(f'out shape: {out.shape}')
out = out.detach().numpy()

# plot result of convolution with a line kernel
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
plt.title('Conv output for all 9 degree rotations')
for i in range(4):
    ax = axes[i // 2, i % 2]
    ax.imshow(out[0, 0, i, :, :], cmap='gray')
plt.tight_layout()
plt.show()

'''
test code from https://github.com/adambielski/GrouPy
# Construct G-Conv layers
C1 = P4ConvZ2(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
C2 = P4ConvP4(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

# Create 10 images with 3 channels and 9x9 pixels:
x = torch.randn(10, 3, 9, 9)
print(type(x[0,0,0,0]))
# fprop
y = C2(C1(x))
print(y.data.shape)  # (10, 64, 4, 9, 9)
'''