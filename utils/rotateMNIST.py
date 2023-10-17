import os.path
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tqdm import tqdm


def generate_rotated_mnist_dataset(root, num_examples=60000, num_rotations=8, max_angle=180):
    """
    :param root:
    :param num_examples: number of mnist images that are used for augmentation
    :param num_rotations: number of times each image is rotated. Use 0 to rotate each image once and not
                        include the original image
    :param max_angle: images are rotated by an angle between [-max_angle, max_angle]
    :return: saves the rotated MNIST dataset in 'datasets/RotMNIST'
    """
    if num_examples > 60000:
        print('Max number of examples is 60000 for MNIST')
        num_examples = 60000

    # load standard MNIST data as numpy array
    dataset = datasets.MNIST(root=root + '/datasets', train=True, transform=transforms.ToTensor(), download=True)
    images, labels = dataset.data.numpy(), dataset.targets.numpy()

    # check data directory
    dir_path = root + '/datasets/RotMNIST'
    if not os.path.isdir(dir_path):
        os.mkdir(path=dir_path)
        print(f'created new directory {dir_path}')

    expanded_images = []
    expanded_labels = []
    # background value. Images are in range [0, 255], 0 is black
    bg_value = 0
    prog_bar = tqdm(total=num_examples * (num_rotations + 1))

    if num_rotations == 0:
        for i in range(num_examples):
            angle = np.random.randint(-max_angle, max_angle, 1)[0]
            new_image = ndimage.rotate(images[i], angle, reshape=False, cval=bg_value)
            expanded_images.append(new_image)
            expanded_labels.append(labels[i])
            prog_bar.update(1)
    else:
        for i in range(num_examples):
            original_image = images[i]
            label = labels[i]
            # append images that are not rotated
            expanded_images.append(original_image)
            expanded_labels.append(label)

            for _ in range(num_rotations):
                angle = np.random.randint(-max_angle, max_angle, 1)[0]
                new_image = ndimage.rotate(original_image, angle, reshape=False, cval=bg_value)

                expanded_images.append(new_image)
                expanded_labels.append(label)

            prog_bar.update(1 + num_rotations)


    prog_bar.close()

    # Display a number of digits and corresponding rotations
    num_display_digits = 4
    if num_rotations == 0:
        num_cols = 8
        start = 0
    else:
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

    # save images and labels
    suffix = f'_{num_examples}ex_{num_rotations}rot_{max_angle}deg.npy'
    np.save(dir_path + '/data' + suffix, expanded_images)
    np.save(dir_path + '/labels' + suffix, expanded_labels)

    print('=' * 32)
    print(f' saved {len(expanded_images)} rotated images and labels')
    print('=' * 32)

#generate_rotated_mnist_dataset(num_examples=12000, num_rotations=0, max_angle=180)
