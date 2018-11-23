import os
from PIL import Image
import numpy as np


def load_swedish_leaf_in_right_format(size):
    path = "./datasets/swedish_leaf"
    classes = os.listdir(path)
    print('All classes: {}'.format(classes))
    images = []
    labels = []
    for idx, class_name in enumerate(classes):
        print("({}/15) Loading {} images...".format(idx + 1, class_name))
        path_to_class_dir = os.path.join(path, class_name)
        image_filenames = os.listdir(path_to_class_dir)
        for filename in image_filenames:
            image_path = os.path.join(path_to_class_dir, filename)
            image = Image.open(image_path)
            image = image.resize(size, Image.ANTIALIAS)
            np_image = np.array(image)
            images.append(np_image)
            labels.append(int(filename.split('nr')[0][1:]))
    return np.array(images), np.array(labels)


size = (32, 32,)
(x, y) = load_swedish_leaf_in_right_format(size)

# Xtrain = [images, height, width, channels]
print(x.shape)
# y_train = [labels]
print(y.shape)

np.save('swedish_leaf{}x{}pix_images'.format(size[0], size[1]), x)
np.save('swedish_leaf{}x{}pix_labels'.format(size[0], size[1]), y)
