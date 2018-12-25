import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import matplotlib
from keras_preprocessing.image import ImageDataGenerator

matplotlib.use('agg')
import matplotlib.pyplot as plt


class AlexNet:
    def __init__(self, data_base_path='data/data', optimizer='adam'):
        # build and compile the network
        self.network = self.build_network()
        self.network.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # load data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data(data_base_path)

        # create a generator
        self.datagen = ImageDataGenerator(
            zoom_range=0.2,  # randomly zoom into images
            rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
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

    def load_data(self, base_path):
        x_train = np.load('{}swedish_leaf64x64pix_train_images.npy'.format(base_path))
        y_train = np.load('{}swedish_leaf64x64pix_train_labels.npy'.format(base_path))-1

        x_test = np.load('{}swedish_leaf64x64pix_test_images.npy'.format(base_path))
        y_test = np.load('{}swedish_leaf64x64pix_test_labels.npy'.format(base_path))-1

        return (x_train, y_train), (x_test, y_test)

    def train_network(self, epochs, create_plots=True, save_model=True):
        history = self.network.fit(self.x_train, self.y_train, epochs=epochs, verbose=2,
                                   validation_data=(self.x_test, self.y_test)) #, shuffle=True
        if create_plots:
            self.plot_accuracy_and_loss(history, epochs)
        if save_model:
            self.save_model(epochs)

    def train_network_with_generator(self, epochs, create_plots=True, save_model=True):
        history = self.network.fit(self.x_train, self.y_train, epochs=epochs, verbose=2,
                                   validation_data=(self.x_test, self.y_test)) #, shuffle=True
        if create_plots:
            self.plot_accuracy_and_loss(history, epochs)
        if save_model:
            self.save_model(epochs)

    def plot_accuracy_and_loss(self, history, epochs):
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("plots/alexnet_accuracy_%d_epochs.png" % epochs)
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("plots/alexnet_loss_%d_epochs.png" % epochs)
        plt.close()

    def save_model(self, epochs):
        model_path = "saved_model/alexnet_%d_epochs.json" % epochs
        weights_path = "saved_model/alexnet_%d_epochs_weights.hdf5" % epochs
        options = {"file_arch": model_path,
                    "file_weight": weights_path}
        json_string = self.network.to_json()
        open(options['file_arch'], 'w').write(json_string)
        self.network.save_weights(options['file_weight'])


np.random.seed(1000)


epochs = 200
lr = 0.000001
decay = lr/epochs
optimizer = keras.optimizers.Adam(lr=lr, decay=decay)
alexnet = AlexNet(data_base_path='../other_GANS/datasets/swedish_np/', optimizer=optimizer)

alexnet.train_network(epochs=epochs)
# filepath = "data/alexnet-cnn.hdf5"

# Checkpoint storing the best checkpoint with improvements for the val_acc
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Train
