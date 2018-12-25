from __future__ import print_function, division

import matplotlib
import numpy as np
from keras.engine.saving import model_from_json

matplotlib.use('agg')
import matplotlib.pyplot as plt


class ImageGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.generator = self.load_model()
        self.amount_of_classes = 15

    def generate_images_for_class(self, image_class, amount):
        noise = np.random.normal(0, 1, (amount, 100))
        sampled_labels = np.array([image_class]*amount)
        return self.generator.predict([noise, sampled_labels])

    def create_png_image_for_class(self, image_class, size=5):
        r, c = size, size
        gen_imgs = self.generate_images_for_class(image_class, r*c)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("shared_images/{}_{}.png".format(self.model_name, image_class))
        plt.close()

    def sample_pictures(self, size=5):
        for image_class in range(self.amount_of_classes):
            self.create_png_image_for_class(image_class, size=size)

    def generate_dataset(self, size_per_class=60):
        # Generate size_per_class images for every class and concatenate the arrays
        images = None
        labels = []
        for image_class in range(self.amount_of_classes):
            if images is None:
                images = self.generate_images_for_class(image_class, size_per_class)
            else:
                images = np.concatenate((images, self.generate_images_for_class(image_class, size_per_class)))
            labels += [image_class] * size_per_class

        # shuffle the dataset
        idx = np.random.permutation(len(labels))
        images, labels = images[idx], np.array(labels)[idx]

        return images, labels

    def save_dataset_as_npy(self, path, size_per_class=60):
        images, labels = self.generate_dataset(size_per_class=size_per_class)
        np.save('{}swedish_leaf_{}_images'.format(path, self.model_name), images)
        np.save('{}swedish_leaf_{}_labels'.format(path, self.model_name), labels)
        return images, labels

    def load_model(self):
        model_path = "saved_model/%s.json" % self.model_name
        weights_path = "saved_model/%s_weights.hdf5" % self.model_name
        # load json and create model
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weights_path)
        print("Loaded {} from disk".format(self.model_name))
        return loaded_model


if __name__ == '__main__':
    NPY_STORAGE = '../datasets/swedish_np/'
    image_generator = ImageGenerator("generator64_50000_2")  # generator64_50000_2, generator64_28000
    images, labels = image_generator.save_dataset_as_npy(path=NPY_STORAGE, size_per_class=60)
    print(images.shape)
    print(labels.shape)
    #image_generator.sample_pictures(size=3)
