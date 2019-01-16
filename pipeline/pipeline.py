from __future__ import print_function, division

import pathlib
import random
import sys
from datetime import datetime

import numpy as np
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
    for i in range(1, 16):
        idxs = np.where(labels == i)
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

    def __init__(self, folder='test'):
        random.seed(datetime.now())
        # Create the folders for results
        pathlib.Path('{}/images'.format(folder)).mkdir(parents=True, exist_ok=True)
        pathlib.Path('{}/augmented/plots'.format(folder)).mkdir(parents=True, exist_ok=True)
        pathlib.Path('{}/original/plots'.format(folder)).mkdir(parents=True, exist_ok=True)
        self.folder = folder + '/'

    def n_runs(self, n=10, **kwargs):
        self.write_props(kwargs)
        for run in range(n):
            self.hashtag_print('Run {}:'.format(run))
            self.run(run_nr=str(run), **kwargs)

    def run(self, split=.8, gan_epochs=30000, alexnet_epochs=300, alexnet_lr=0.00001, alexnet_decay=0, run_nr='0', only_generated=False):
        """
        1. Split the data according to a float split
        2. Train the ACGAN on the training data for gan_epochs amount of epochs, 1 sample image is saved at the end of
           training.
        3. Generate extra training data (size of this data is equal to the size of the original training data)
        4. Train the AlexNet with only the original data and save plots and csv of the training process
        5. Train the AlexNet with original data and the generated extra training data
        """

        # Split the data
        self.hashtag_print('Splitting data with a {} split.'.format(split))
        x_train, y_train, x_test, y_test = split_data(split=split)

        # Train an ACGAN on the training data
        self.hashtag_print('Training ACGAN for {} epochs.'.format(gan_epochs))
        gan = self.train_gan(x_train, np.copy(y_train), gan_epochs, self.folder, run_nr)

        # Generate extra training data with the GAN
        self.hashtag_print('Generating extra training data using the trained ACGAN.')
        x_train_generated, y_train_generated = gan.generate_dataset(size_per_class=200 if only_generated
                                                                    else int(x_train.shape[0]/15)*2,
                                                                    replace_collapsed_classes=(not only_generated))
        gan.delete()

        # Train on only original dataset
        if not only_generated:
            self.hashtag_print('Training AlexNet on original training data.')
            self.train_alexnet(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, lr=alexnet_lr,
                               decay=alexnet_decay,
                               epochs=alexnet_epochs, folder='{}original/'.format(self.folder), run_nr=run_nr)

        # Train on augmented dataset
        self.hashtag_print('Training AlexNet on augmented data.')
        self.train_alexnet(x_train=x_train_generated if only_generated else np.concatenate((x_train, x_train_generated)),
                           y_train=y_train_generated if only_generated else np.concatenate((y_train, y_train_generated)),
                           x_test=x_test, y_test=y_test, lr=alexnet_lr, decay=alexnet_decay, epochs=alexnet_epochs,
                           folder='{}augmented/'.format(self.folder), run_nr=run_nr)
        del x_train_generated
        del y_train_generated
        del x_train
        del y_train
        del x_test
        del y_test

    @staticmethod
    def train_gan(x_train, y_train, epochs, folder, run_nr):
        gan = ACGAN(x_train, y_train, folder, run_nr=run_nr, print_intermediate_images=False)
        gan.train(epochs=epochs, batch_size=32)
        return gan

    @staticmethod
    def train_alexnet(x_train, y_train, x_test, y_test, epochs, lr, decay, folder, run_nr):
        alexnet = AlexNet(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, lr=lr, decay=decay,
                          folder=folder, run_nr=run_nr)
        alexnet.train_network_with_generator(epochs=epochs, save_model=False)
        alexnet.delete()

    @staticmethod
    def hashtag_print(string):
        print('#'*50, '\n{}\n'.format(string), '#'*50)

    def write_props(self, d):
        with open('{}props'.format(self.folder), 'w') as f:
            f.writelines(["{}: {}\n".format(key, value) for key, value in d.items()])
            f.flush()


if __name__ == '__main__':
    split = float(sys.argv[1])
    pipe = Pipeline(folder="{}_split_{}".format(datetime.now().strftime("%d_%m_%Y"), str(split).replace('.', '')))
    pipe.n_runs(n=6, split=split, gan_epochs=50000, alexnet_epochs=600,  alexnet_lr=0.0001, alexnet_decay=0.05)
