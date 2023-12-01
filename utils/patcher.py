import numpy as np
import matplotlib.pyplot as plt
import h5py
import patchify


def unpatch(patches, stride=1):
    p = patches.shape[-1]
    im_h = (patches.shape[0] - 1) * stride + p
    im_w = (patches.shape[1] - 1) * stride + p

    reconstructed = np.zeros((im_h, im_w))
    div = np.zeros((im_h, im_w))

    for x in range(patches.shape[0]):
        for y in range(patches.shape[1]):
            x_ = x * stride
            y_ = y * stride
            reconstructed[x_:x_ + p, y_:y_ + p] += patches[x, y]
            div[x_:x_ + p, y_:y_ + p] += np.ones((p, p))

    return reconstructed / div


def unpatch_test():
    file_path = '../datasets/LoDoPaB/ground_truth_train/ground_truth_train_000.hdf5'
    f = h5py.File(file_path, 'r')
    dataset = f['data']
    im = dataset[0]
    print(f'image shape: {im.shape}')
    N = im.shape[0]
    P = 8
    stride = 1
    patches = patchify.patchify(dataset[0], patch_size=(P, P), step=stride)
    print(f'patches shape: {patches.shape}')

    recon = unpatch(patches, stride=stride)
    print(f'recon.shape: {recon.shape}')
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(im)
    ax[0].set_title('original')
    ax[1].imshow(recon)
    ax[1].set_title('unpatched')
    plt.show()
