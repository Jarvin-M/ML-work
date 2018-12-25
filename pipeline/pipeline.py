import random

import numpy as np
from datetime import datetime

from acgan64_for_pipeline import ACGAN
from alexnet64_thijs import AlexNet


def split_class(images, labels, split):
    # Shuffle the arrays
    zipped = list(zip(images, labels))
    random.shuffle(zipped)
    images, labels = zip(*zipped)

    # Split the arrays
    split_idx = int(len(labels) * split)
    return images[:split_idx], labels[:split_idx], images[split_idx:], labels[split_idx:]


def split_data(split=.8):
    images = np.load('../other_GANS/datasets/swedish_np/swedish_leaf64x64pix_all_images.npy')
    labels = np.load('../other_GANS/datasets/swedish_np/swedish_leaf64x64pix_all_labels.npy')
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(1,16):
        idxs = np.where(labels==i)
        (x_train_temp, y_train_temp, x_test_temp, y_test_temp) = split_class(images[idxs], labels[idxs], split)
        x_train.extend(x_train_temp)
        y_train.extend(y_train_temp)
        x_test.extend(x_test_temp)
        y_test.extend(y_test_temp)

    # Final shuffle of the arrays
    idx = np.random.permutation(len(y_train))
    x_train, y_train = np.array(x_train)[idx], np.array(y_train)[idx]
    idx = np.random.permutation(len(y_test))
    x_test, y_test = np.array(x_test)[idx], np.array(y_test)[idx]

    # Configure inputs
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5

    # y_train ranges from 1-15 should 0-14
    y_train -= 1
    y_test -= 1

    return x_train, y_train, x_test, y_test


class Pipeline:
    def __init__(self, split=.8, gan_epochs=30000, alexnet_epochs=300,  alexnet_lr=0.00001, folder='run1/'):
        random.seed(datetime.now())
        self.folder = folder

        # Split the data
        x_train, y_train, x_test, y_test = split_data(split=split)

        # Create extra image of the training data with a GAN
        x_train_generated, y_train_generated = self.generate_data(x_train, np.copy(y_train), gan_epochs)

        # Train on only original dataset
        print('#'*50, '\nTraining AlexNet on original data\n', '#'*50)
        original = AlexNet(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, lr=alexnet_lr,
                           folder='{}original/'.format(self.folder))
        original.train_network_with_generator(epochs=alexnet_epochs, save_model=False)

        # Train on augmented dataset
        print('#'*50, '\nTraining AlexNet on augmented data\n', '#'*50)
        augmented = AlexNet(x_train=np.concatenate((x_train, x_train_generated)),
                            y_train=np.concatenate((y_train, y_train_generated)),
                            x_test=x_test, y_test=y_test, lr=alexnet_lr,
                            folder='{}augmented/'.format(self.folder))
        augmented.train_network_with_generator(epochs=alexnet_epochs, save_model=False)

    def generate_data(self, x_train, y_train, epochs):
        print('#'*50, '\nTraining GAN and generating dataset\n', '#'*50)
        gan = ACGAN(x_train, y_train, self.folder)
        gan.train(epochs=epochs, batch_size=32)
        return gan.generate_dataset(size_per_class=int(x_train.shape[0]/15))


Pipeline(split=0.2, folder="run1/")

# Split the data (0.2 - 0.8) ?
# Train gan with training data
# Generate extra dataset
# Train alexnet trainging data
# Train alexnet with training data and generated data