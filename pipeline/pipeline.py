from __future__ import print_function, division

import random

import keras
import matplotlib
import numpy as np
from datetime import datetime
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Embedding, ZeroPadding2D
from keras.layers import Input, Reshape, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

matplotlib.use('agg')
import matplotlib.pyplot as plt


class AlexNet:
    def __init__(self, x_train, y_train, x_test, y_test, lr=0.00001, folder=''):
        self.folder = folder
        self.lr = lr
        # build and compile the network
        self.network = self.build_network()
        optimizer = keras.optimizers.Adam(lr=self.lr)
        self.network.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # load data
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # create a generator to transform the pictures
        self.datagen = ImageDataGenerator(
            zoom_range=0.05,  # randomly zoom into images
            rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.07,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.07,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False  # randomly flip images
        )

    def build_network(self):
        # A sequential alexnet
        alexnet = Sequential()

        # 1st Convolutional Layer
        alexnet.add(Conv2D(filters=32, input_shape=(64, 64, 3), kernel_size=(4, 4), padding='same'))
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # Batch Normalisation before passing it to the next layer
        alexnet.add(BatchNormalization())

        # 2nd Convolutional Layer
        alexnet.add(Conv2D(filters=96, kernel_size=(4, 4), strides=(1, 1), padding='same'))
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        alexnet.add(BatchNormalization())

        # 3rd Convolutional Layer
        alexnet.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), padding='same'))
        alexnet.add(Activation('relu'))
        alexnet.add(BatchNormalization())

        # 4th Convolutional Layer
        alexnet.add(Conv2D(filters=256, kernel_size=(4, 4), strides=(1, 1), padding='same'))
        alexnet.add(Activation('relu'))
        alexnet.add(BatchNormalization())

        # 5th Convolutional Layer
        alexnet.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), padding='same'))
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        alexnet.add(BatchNormalization())

        # Passing it to a dense layer- full connected layer
        alexnet.add(Flatten())

        # 1st Dense Layer
        alexnet.add(Dense(4096, input_shape=(32 * 32 * 3,)))
        alexnet.add(Activation('relu'))
        alexnet.add(Dropout(0.4))
        alexnet.add(BatchNormalization())

        # 2nd Dense Layer
        alexnet.add(Dense(4096))
        alexnet.add(Activation('relu'))
        alexnet.add(Dropout(0.4))
        alexnet.add(BatchNormalization())

        # 3rd Dense Layer
        alexnet.add(Dense(1000))
        alexnet.add(Activation('relu'))
        alexnet.add(Dropout(0.4))
        alexnet.add(BatchNormalization())

        # Output Layer
        alexnet.add(Dense(15))
        alexnet.add(Activation('softmax'))

        alexnet.summary()

        return alexnet

    def train_network(self, epochs, create_plots=True, save_model=True):
        history = self.network.fit(self.x_train, self.y_train, epochs=epochs, verbose=2,
                                   validation_data=(self.x_test, self.y_test), shuffle=True)
        if create_plots:
            self.plot_accuracy_and_loss(history, epochs)
        if save_model:
            self.save_model(epochs)
        return history

    def train_network_with_generator(self, epochs, create_plots=True, save_model=True):
        history = self.network.fit_generator(self.datagen.flow(self.x_train, self.y_train, batch_size=32),
                                             steps_per_epoch=int(np.ceil(900 / float(32))),
                                             epochs=epochs, verbose=2,
                                             validation_data=(self.x_test, self.y_test), shuffle=True)
        if create_plots:
            self.plot_accuracy_and_loss(history, epochs)
        if save_model:
            self.save_model(epochs)
        return history

    def plot_accuracy_and_loss(self, history, epochs):
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy {}'.format(self.lr))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("{}plots/alexnet_accuracy_{}_epochs_{}.png".format(self.folder, epochs, self.lr))
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss {}'.format(self.lr))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("{}plots/alexnet_loss_{}_epochs_{}.png".format(self.folder, epochs, self.lr))
        plt.close()

    def save_model(self, epochs):
        model_path = "saved_model/alexnet_%d_epochs.json" % epochs
        weights_path = "saved_model/alexnet_%d_epochs_weights.hdf5" % epochs
        options = {"file_arch": model_path,
                    "file_weight": weights_path}
        json_string = self.network.to_json()
        open(options['file_arch'], 'w').write(json_string)
        self.network.save_weights(options['file_weight'])

    def sample_transformed_x(self):
        transformed = self.datagen.random_transform(self.x_train[300])
        plt.imshow(transformed)
        plt.savefig('example.png')
        plt.close()


class ACGAN():
    def __init__(self, x_train, y_train, folder=''):
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

        optimizer = Adam(0.0002, 0.5)
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

        model.summary()

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
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes+1, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128):
        self.y_train = self.y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, self.x_train.shape[0], batch_size)
            imgs = self.x_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, 15, (batch_size, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels. 0-14 if image is valid or 15 if it is generated (fake)
            img_labels = self.y_train[idx]
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

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))

        self.sample_images(epochs)

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
        fig.savefig("%simages/%d_64.png" % (self.folder, epoch))
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