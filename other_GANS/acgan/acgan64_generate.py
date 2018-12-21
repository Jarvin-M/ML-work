from __future__ import print_function, division

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.engine.saving import model_from_json


class ImageGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.generator = self.load_model()

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
    image_generator = ImageGenerator("generator64_50000")
    for image_class in range(14):
        image_generator.create_png_image_for_class(image_class, size=3)
