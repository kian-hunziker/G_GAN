import os.path
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def generate_rotated_mnist_dataset(dir_path='../datasets/RotMNIST',
                                   num_examples=60000,
                                   num_rotations=8,
                                   max_angle=180):

    # load standard MNIST data as numpy array
    dataset = datasets.MNIST(root='../datasets', train=True, transform=transforms.ToTensor(), download=True)
    images, labels = dataset.data.numpy(), dataset.targets.numpy()

    # check data directory
    if not os.path.isdir(dir_path):
        os.mkdir(path=dir_path)
        print(f'created new directory {dir_path}')

    expanded_images = []
    expanded_labels = []

    # background value. Images are in range [0, 255], 0 is black
    bg_value = 0

    for i in range(num_examples):
        original_image = images[i]
        label = labels[i]
        # append images that are not rotated
        expanded_images.append(original_image)
        expanded_labels.append(label)

        if i % 500 == 0:
            print(f'rotated {i} / {num_examples} digits')

        for _ in range(num_rotations):
            angle = np.random.randint(-max_angle, max_angle, 1)[0]
            new_image = ndimage.rotate(original_image, angle, reshape=False, cval=bg_value)

            expanded_images.append(new_image)
            expanded_labels.append(label)

    # save images and labels
    np.save(dir_path + '/images.npy', expanded_images)
    np.save(dir_path + '/labels.npy', expanded_labels)

    print('=' * 16)
    print(f' saved {len(expanded_images)} rotated images and labels')
    print('=' * 16)

    # Display a number of digits and corresponding rotations
    num_display_digits = 4
    num_cols = num_rotations + 1
    start = np.random.randint(0, num_examples) * num_cols
    fig, axes = plt.subplots(num_display_digits, num_cols, figsize=(12, 4))
    for i in range(num_display_digits * num_cols):
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(expanded_images[start + i], cmap='gray')
        ax.axis('off')
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()

