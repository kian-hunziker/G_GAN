import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import h5py
from patchify import patchify

from utils.siren_utils import get_mgrid


class LodopabDataset(torch.utils.data.Dataset):
    """
    Dataset to load patches of LoDoPaB images
    """
    def __init__(self, file_path: str, patch_size: int = 8, num_images: int = 128, print_output: bool = True):
        """
        Initialize dataset. Data is loaded, split into patches and transformed to a torch.tensor
        :param file_path: path of h5 dataset
        :param patch_size: size of patches. Patches are quadratic [patch_size, patch_size]
        :param num_images: number of images to load from the dataset
        :param print_output: If True, a brief summary is printed to the console
        """
        f = h5py.File(file_path, 'r')
        images = f['data'][:]
        self.img_size = images.shape[-1]
        self.patch_size = patch_size
        assert num_images <= images.shape[0]
        self.num_images = num_images
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


class PatchedImage(Dataset):
    """
    Dataset to load and patchify a single image.
    """
    def __init__(self, img_path: str, img_idx: int = 0, patch_size: int = 8,
                 noise_strength: int = 0, use_grid_sample: bool = False):
        """
        Initialize dataset.
        :param img_path: Path to the h5 dataset
        :param img_idx: Index of the image to load
        :param patch_size: Size of the patches to generate. Patches are quadratic
        :param use_grid_sample: If enabled, the get_item method generates a random 2D coordinate in the range
        [-1, 1] for both x and y. Then affine_grid() and grid_sample() are used to sample a patch from the image.
        """
        super().__init__()
        # load single image
        f = h5py.File(img_path)
        img = f['data'][img_idx]

        assert img.shape[-1] == img.shape[-2]

        # add noise to image
        img = img + noise_strength * (np.random.randn(img.shape[-2], img.shape[-1]).astype(np.single) + 0.5)
        self.img = img

        self.use_grid_sample = use_grid_sample
        self.patch_size = patch_size
        self.num_patches = (self.img.shape[-2] - self.patch_size + 1) * (self.img.shape[-1] - self.patch_size + 1)

        if use_grid_sample is True:
            self.img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).type(torch.float)
            coord_range_x = (img.shape[-1] - patch_size) / img.shape[-1]
            coord_range_y = (img.shape[-2] - patch_size) / img.shape[-2]
            self.coord_range = torch.tensor([coord_range_x, coord_range_y])
            scale_width = patch_size / img.shape[-1]
            scale_height = patch_size / img.shape[-2]
            self.scale_matrix = torch.tensor([[scale_width, 0], [0, scale_height]])
        else:
            # generate patches
            self.patches = patchify(img, patch_size=(patch_size, patch_size), step=1)

            # generate 2D coords in range [-1, 1]
            self.coords = get_mgrid(self.patches.shape[0], dim=2)
            # reverse order of y-coords. The top left corner should have coords [-1, 1]
            temp_x = 1.0 * self.coords[:, 0]
            self.coords[:, 0] = 1.0 * self.coords[:, 1]
            self.coords[:, 1] = 1.0 * temp_x
            self.coords[:, 1] = -1.0 * self.coords[:, 1]

    def grid_sample_patch(self, coord: torch.Tensor):
        coord_affine = coord * self.coord_range
        coord_affine[1] = -1.0 * coord_affine[1]
        theta = torch.hstack((self.scale_matrix, coord_affine.unsqueeze(-1))).unsqueeze(0)
        aff_grid = F.affine_grid(theta, [1, 1, self.patch_size, self.patch_size])
        sampled_patch = F.grid_sample(self.img, aff_grid, align_corners=False)
        return sampled_patch[0]

    def get_all_coords_and_patches(self):
        coords = get_mgrid(self.img.shape[-1] - self.patch_size + 1, 2)
        temp_x = 1.0 * coords[:, 0]
        coords[:, 0] = 1.0 * coords[:, 1]
        coords[:, 1] = 1.0 * temp_x
        coords[:, 1] = -1.0 * coords[:, 1]
        if self.use_grid_sample is True:
            im = self.img.squeeze().numpy()
        else:
            im = self.img
        patches = patchify(im, patch_size=(self.patch_size, self.patch_size), step=1)
        return coords, patches

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        if self.use_grid_sample is True:
            coord = 2.0 * (torch.rand(2) - 0.5)
            patch = self.grid_sample_patch(coord)
            return coord, patch
        else:
            x_idx = idx // self.patches.shape[0]
            y_idx = idx % self.patches.shape[1]
            return self.coords[idx], self.patches[x_idx, y_idx]
