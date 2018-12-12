import os
from PIL import Image
import numpy as np
import random
from datetime import datetime


def read_and_scale_image(image_path, size):
    image = Image.open(image_path)
    image = image.resize(size, Image.ANTIALIAS)
    return np.array(image)


def load_one_class(class_path, size, split):
    image_filenames = os.listdir(class_path)

    # Load images and labels into arrays
    images = [read_and_scale_image(os.path.join(class_path, filename), size) for filename in image_filenames]
    labels = [int(filename.split('nr')[0][1:]) for filename in image_filenames]

    # Shuffle the arrays
    zipped = list(zip(images, labels))
    random.shuffle(zipped)
    images, labels = zip(*zipped)

    # Split the arrays
    split_idx = int(len(labels) * split)
    return images[:split_idx], labels[:split_idx], images[split_idx:], labels[split_idx:]


def load_swedish_leaf_in_right_format(path, size, split):
    classes = os.listdir(path)
    print('All classes: {}'.format(classes))
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for idx, class_name in enumerate(classes):
        print("({}/15) Loading {} images...".format(idx + 1, class_name))
        (x_train_temp, y_train_temp, x_test_temp, y_test_temp) = load_one_class(os.path.join(path, class_name), size, split)
        x_train.extend(x_train_temp)
        y_train.extend(y_train_temp)
        x_test.extend(x_test_temp)
        y_test.extend(y_test_temp)

    # Final shuffle of the arrays
    idx = np.random.permutation(len(y_train))
    x_train, y_train = np.array(x_train)[idx], np.array(y_train)[idx]
    idx = np.random.permutation(len(y_test))
    x_test, y_test = np.array(x_test)[idx], np.array(y_test)[idx]

    return x_train, y_train, x_test, y_test


# Parameters
DATADIR = './datasets/swedish_leaf'
NPY_STORAGE = '../other_GANS/datasets/swedish_np/'
size = (64, 64,)
split = .8

# Run the functions above...
random.seed(datetime.now())
(x_train, y_train, x_test, y_test) = load_swedish_leaf_in_right_format(DATADIR, size, split)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

np.save('{}swedish_leaf{}x{}pix_train_images'.format(NPY_STORAGE, size[0], size[1]), x_train)
np.save('{}swedish_leaf{}x{}pix_train_labels'.format(NPY_STORAGE, size[0], size[1]), y_train)
np.save('{}swedish_leaf{}x{}pix_test_images'.format(NPY_STORAGE, size[0], size[1]), x_test)
np.save('{}swedish_leaf{}x{}pix_test_labels'.format(NPY_STORAGE, size[0], size[1]), y_test)
