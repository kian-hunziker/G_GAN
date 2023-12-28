import random

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
    def __init__(self, file_path: str, patch_size: int = 8, num_images: int = 128, print_output: bool = True,
                 use_grid_sample: bool = False):
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
        self.use_grid_sample = use_grid_sample

        # normalize each image separately
        normalized_images = []
        for i in range(num_images):
            img = f['data'][i]
            min_val = np.min(img)
            max_val = np.max(img)
            normalized_images.append((img - min_val) / (max_val - min_val))

        if use_grid_sample is True:
            # convert numpy images with dimension [height, width] to torch tensors [1, 1, height, width]
            self.tensor_images_normalized = [
                torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).type(torch.float32)
                for norm_img in normalized_images
            ]

            # prepare for affine grid
            # coord range is needed to make sure we do not go beyond the actual image
            height = images.shape[-2]
            width = images.shape[-1]
            coord_range_x = (width - patch_size) / width
            coord_range_y = (height - patch_size) / height
            self.coord_range = torch.tensor([coord_range_x, coord_range_y])

            # scale_matrix is a diagonal matrix that is used to adjust the image to the size of the patch we sample
            scale_width = patch_size / width
            scale_height = patch_size / height
            self.scale_matrix = torch.tensor([[scale_width, 0], [0, scale_height]])
        else:
            # patchify the images and store the patches
            self.patched_images = [
                patchify(img, patch_size=(self.patch_size, self.patch_size), step=1).reshape(-1, self.patch_size,
                                                                                                 self.patch_size)
                for img in normalized_images
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
            if self.use_grid_sample is True:
                print('Using grid sample')
            else:
                print('Using patchify, not grid sample')
                print(f'Number of patches per image: {self.num_patches_per_image}')
                print(f'Total number of patches: {self.total_num_patches}')
            print('-' * 32 + '\n')

    def __getitem__(self, index):
        """
        Returns a patch with dimensions [1, patch_size, patch_size]. If use_grid_sample is enabled, the index
        will be ignored. Instead, we generate a random coordinate and use affine_grid in conjunction with grid_sample
        to extract a patch from the original images.
        If use_grid_sample is not enabled, the index will be converted to an img_idx and a patch index and
        the corresponding patch will be looked up and returned
        :param index: index of patch
        :return: Patch with dimension [1, patch_size, patch_size] as torch.Tensor
        """
        if self.use_grid_sample is True:
            # select random image
            img_idx = random.randint(0, self.num_images - 1)

            # generate random coordinates and scale them
            coords = 2.0 * (torch.rand(2) - 0.5) * self.coord_range

            # define input affine matrix [1 x 2 x 3]
            theta = torch.hstack((self.scale_matrix, coords.unsqueeze(-1))).unsqueeze(0)

            # generate affine grid
            aff_grid = F.affine_grid(theta, [1, 1, self.patch_size, self.patch_size])

            # sample from random image
            sampled_patch = F.grid_sample(self.tensor_images_normalized[img_idx], aff_grid, align_corners=False)
            return sampled_patch.squeeze(dim=0)
        else:
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
                 noise_strength: int = 0, use_grid_sample: bool = False,
                 normalize: bool = False):
        """
        Initialize dataset.
        :param img_path: Path to the h5 dataset
        :param img_idx: Index of the image to load
        :param patch_size: Size of the patches to generate. Patches are quadratic
        :param use_grid_sample: If enabled, the get_item method generates a random 2D coordinate in the range
        [-1, 1] for both x and y. Then affine_grid() and grid_sample() are used to sample a patch from the image.
        :param normalize: if true normalize the image to range [0, 1]
        """
        super().__init__()
        # load single image
        f = h5py.File(img_path)
        img = f['data'][img_idx]

        assert img.shape[-1] == img.shape[-2]

        # normalize image
        if normalize:
            min_val = np.min(img)
            max_val = np.max(img)
            img = (img - min_val) / (max_val - min_val)

        # add noise to image
        img = img + noise_strength * (np.random.randn(img.shape[-2], img.shape[-1]).astype(np.single))
        self.img = img

        self.use_grid_sample = use_grid_sample
        self.patch_size = patch_size
        self.num_patches = (self.img.shape[-2] - self.patch_size + 1) * (self.img.shape[-1] - self.patch_size + 1)

        if use_grid_sample is True:
            self.img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).type(torch.float32)
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
        self.coords = get_mgrid(self.img.shape[-1] - self.patch_size + 1, 2)
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
        return sampled_patch.squeeze()

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
            # placeholder to test if training difficulties are caused by continuous patches
            # coord = self.coords[idx]
            patch = self.grid_sample_patch(coord)
            return coord, patch
        else:
            x_idx = idx // self.patches.shape[0]
            y_idx = idx % self.patches.shape[1]
            return self.coords[idx], self.patches[x_idx, y_idx]
