import torch
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# Create a 28x28 array filled with -1s
square_img = -1 * np.ones((28, 28))

# Define the dimensions of the central 7x7 square
square_size = 15

# Calculate the starting and ending indices for the square
start_row = (square_img.shape[0] - square_size) // 2
end_row = start_row + square_size
start_col = (square_img.shape[1] - square_size) // 2
end_col = start_col + square_size

# Fill the edges of the central square with ones
square_img[start_row, start_col:end_col] = 1
square_img[end_row - 1, start_col:end_col] = 1
square_img[start_row:end_row, start_col] = 1
square_img[start_row:end_row, end_col - 1] = 1

# rotate square
rot_square = ndimage.rotate(square_img, 45, reshape=False, cval=-1)

# Print the resulting array
fig, axes = plt.subplots(1, 2)
axes[0].imshow(square_img, cmap='gray')
axes[1].imshow(rot_square, cmap='gray')
plt.title('input')
plt.show()

square_input_tensor = torch.from_numpy(square_img).unsqueeze(0).unsqueeze(0).float()
rot_input_tensor = torch.from_numpy(rot_square).unsqueeze(0).unsqueeze(0).float()
print(f'array_tensor.shape: {square_input_tensor.shape}')

conv_line_kernel = P4ConvZ2(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
conv_diag_kernel = P4ConvZ2(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
rand_kernel = conv_line_kernel.weight[0][0][0].detach().numpy()

line_kernel = np.zeros((3, 3))
line_kernel[1] = np.ones(3)

diag_kernel = np.zeros((3, 3))
for i in range(3):
    diag_kernel[i, i] = 1

# plot kernels
fig, axes = plt.subplots(1, 2)
axes[0].imshow(line_kernel, cmap='gray')
axes[1].imshow(diag_kernel, cmap='gray')
plt.tight_layout()
plt.title('line kernel left, diag kernel right')
plt.show()

# set conv kernel to line
line_kernel_torch = torch.from_numpy(line_kernel).float()
line_kernel_torch.requires_grad = True
diag_kernel_torch = torch.from_numpy(diag_kernel).float()
diag_kernel_torch.requires_grad = True
with torch.no_grad():
    conv_line_kernel.weight[0][0][0] = line_kernel_torch
    conv_diag_kernel.weight[0][0][0] = diag_kernel_torch

out_line = conv_line_kernel(square_input_tensor)
out_diag = conv_diag_kernel(rot_input_tensor)
print(f'out shape: {out_line.shape}')
out_line = out_line.detach().numpy()
out_diag = out_diag.detach().numpy()

# plot result of convolution with a line kernel
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
plt.title('Conv output for all 9 degree rotations')
for i in range(4):
    ax = axes[i // 2, i % 2]
    ax.imshow(out_line[0, 0, i, :, :], cmap='gray')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
plt.title('Conv output for all 9 degree rotations')
for i in range(4):
    ax = axes[i // 2, i % 2]
    ax.imshow(out_diag[0, 0, i, :, :], cmap='gray')
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