%%time
import keras
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
import numpy as np

np.random.seed(1000)

x_train = np.load('data/dataswedish_leaf64x64pix_train_images.npy')
y_train = np.load('data/dataswedish_leaf64x64pix_train_labels.npy')-1

x_test = np.load('data/dataswedish_leaf64x64pix_test_images.npy')
y_test = np.load('data/dataswedish_leaf64x64pix_test_labels.npy')-1



# A sequential alexnet
alexnet = Sequential()

# 1st Convolutional Layer
alexnet.add(Conv2D(filters=32, input_shape=(64,64,3), kernel_size=(4,4), padding='same'))
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2,2), padding='same'))
# Batch Normalisation before passing it to the next layer
alexnet.add(BatchNormalization())

# 2nd Convolutional Layer
alexnet.add(Conv2D(filters=96, kernel_size=(4,4), strides=(1,1), padding='same'))
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
alexnet.add(BatchNormalization())

# 3rd Convolutional Layer
alexnet.add(Conv2D(filters=128, kernel_size=(4,4), strides=(1,1), padding='same'))
alexnet.add(Activation('relu'))
alexnet.add(BatchNormalization())

# 4th Convolutional Layer
alexnet.add(Conv2D(filters=256, kernel_size=(4,4), strides=(1,1), padding='same'))
alexnet.add(Activation('relu'))
alexnet.add(BatchNormalization())

# 5th Convolutional Layer
alexnet.add(Conv2D(filters=128, kernel_size=(4,4), strides=(1,1), padding='same'))
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
alexnet.add(BatchNormalization())

# Passing it to a dense layer- full connected layer
alexnet.add(Flatten())

# 1st Dense Layer
alexnet.add(Dense(4096, input_shape=(32*32*3,)))
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

# Compile 
# sgd = optimizers.SGD(lr=0.01, decay=0.0, momentum=0.1, nesterov=False)
# model.compile(loss='mean_squared_error', optimizer=)
alexnet.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

filepath = "data/alexnet-cnn.hdf5"

# Chepoint storing the best checkpoint with improvements for the val_acc
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Train
alexnet.fit(x_train, y_train, epochs=4, verbose=1, validation_data=(x_test,y_test), shuffle=True, callbacks=[checkpoint])