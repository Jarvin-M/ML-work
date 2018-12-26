from __future__ import print_function, division

import matplotlib
import numpy as np
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

matplotlib.use('agg')
import matplotlib.pyplot as plt


class ACGAN():
    def __init__(self, x_train, y_train, folder='', run_nr=''):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 15
        self.latent_dim = 100

        self.x_train = x_train
        self.y_train = y_train
        self.folder = folder
        self.run_nr = run_nr

        self.lr = 0.0001
        optimizer = Adam(self.lr, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(512 * 4 * 4, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((4, 4, 512)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(512, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        # model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        # model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes+1, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128):
        self.y_train = self.y_train.reshape(-1, 1)  # [1, 0, 14] -> [[1], [0], [14]]

        datagenerator = ImageDataGenerator(
            rotation_range=4,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.02,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.02,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False  # randomly flip images
        )
        datagen_iterator = datagenerator.flow(self.x_train, self.y_train, batch_size=batch_size, shuffle=True)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        history = {'d_loss': [], 'g_loss': [], 'acc': [], 'op_acc': []}

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images and corresponding labels
            imgs, img_labels = datagen_iterator.next()
            while img_labels.shape[0] != batch_size:  # Sometimes something goes wrong with the batch_size
                imgs, img_labels = datagen_iterator.next()

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, 15, (batch_size, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels. 0-14 if image is valid or 15 if it is generated (fake)
            fake_labels = 15*np.ones(img_labels.shape)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            history['d_loss'].append(d_loss[0])
            history['g_loss'].append(g_loss[0])
            history['acc'].append(d_loss[3])
            history['op_acc'].append(d_loss[4])

            # Plot the progress
            if (epoch+1) % 1000 == 0 or epochs < 10:
                print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
            if epochs > 10 and epoch % (epochs/4) == 0:
                self.sample_images(epoch)

        self.sample_images(epochs)
        self.plot_accuracy_and_loss(history, epochs)

    def sample_images(self, epoch):
        r, c = 10, 15
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("%simages/%d_run_%s.png" % (self.folder, epoch, self.run_nr))
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "%ssaved_model/%s.json" % (self.folder, model_name)
            weights_path = "%ssaved_model/%s_weights.hdf5" % (self.folder, model_name)
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator64")
        save(self.discriminator, "discriminator64")

    def plot_accuracy_and_loss(self, history, epochs):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        # summarize history for accuracy
        ax1.plot(history['acc'])
        ax1.plot(history['op_acc'])
        ax1.axis(xmin=0, xmax=epochs-1, ymin=0, ymax=1)
        ax1.set_title('model accuracy')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.legend(['acc', 'op_acc'], loc='upper left')

        # summarize history for loss
        ax2.plot(history['d_loss'])
        ax2.plot(history['g_loss'])
        ax2.axis(xmin=0, xmax=epochs-1)
        ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['d_loss', 'g_loss'], loc='upper left')

        fig.suptitle('ACGAN with learning rate {}'.format(self.lr))
        fig.savefig("{}images/acgan_accuracy_loss_{}_epochs_{}_lr_{}.png".format(self.folder, epochs, self.run_nr,
                                                                                 self.lr))

    def generate_images_for_class(self, image_class, amount):
        noise = np.random.normal(0, 1, (amount, 100))
        sampled_labels = np.array([image_class]*amount)
        return self.generator.predict([noise, sampled_labels])

    def generate_dataset(self, size_per_class=60):
        # Generate size_per_class images for every class and concatenate the arrays
        images = None
        labels = []
        for image_class in range(self.num_classes):
            if images is None:
                images = self.generate_images_for_class(image_class, size_per_class)
            else:
                images = np.concatenate((images, self.generate_images_for_class(image_class, size_per_class)))
            labels += [image_class] * size_per_class

        # shuffle the dataset
        idx = np.random.permutation(len(labels))
        images, labels = images[idx], np.array(labels)[idx]

        return images, labels


if __name__ == '__main__':
    from pipeline import split_data
    x_train, y_train, _, _ = split_data(.1)
    acgan = ACGAN(x_train, y_train)
    acgan.train(epochs=10000, batch_size=32)
