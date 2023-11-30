import h5py
import torch
import torchvision.transforms as transforms
from patchify import patchify


class LodopabDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, patch_size: int = 8, print_output: bool = True):
        f = h5py.File(file_path, 'r')
        images = f['data'][:]
        self.img_size = images.shape[-1]
        self.patch_size = patch_size
        self.num_images = images.shape[0]
        self.num_patches_per_image = (self.img_size - self.patch_size + 1) ** 2
        self.total_num_patches = self.num_patches_per_image * self.num_images

        self.patched_images = [
            patchify(
                f['data'][i], patch_size=(self.patch_size, self.patch_size), step=1).reshape(-1, self.patch_size,
                                                                                             self.patch_size)
            for i in range(self.num_images)
        ]

        self.transform = transforms.Compose(
            [transforms.ToTensor(), ]
        )

        if print_output is True:
            print('-' * 32)
            print(f'Loaded LoDoPaB data from: {file_path}')
            print(f'Number of images: {self.num_images}')
            print(f'Image size: ({self.img_size}, {self.img_size})')
            print(f'Patch size: {self.patch_size}')
            print(f'Number of patches per image: {self.num_patches_per_image}')
            print(f'Total number of patches: {self.total_num_patches}')
            print('-' * 32 + '\n')

    def __getitem__(self, index):
        img_idx = index // self.num_patches_per_image
        patch_idx = index % self.num_patches_per_image
        x = self.patched_images[img_idx][patch_idx]
        x = self.transform(x)
        return x

    def __len__(self):
        return self.total_num_patches
