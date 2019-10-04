"""
Convolutional Neural Netwok implementation in tensorflow whith fully connected layers.

The neural network is ran against the mnist dataset and we can see an example of distortion of input in the case
where the input comes from memory.
"""
import tempfile

import tensorflow as tf
import os
import time
from pathlib import Path
import urllib.request
import numpy as np
import keras
import scipy.io as sio

from keras import Sequential
from keras.initializers import he_normal
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator

def download_data(url, directory, name=None):
    """
    Download the file at the specified url

    :param url: the end-point url of the need file
    :type url: str
    :param directory: the target directory where to download the file
    :type directory: str
    :param name: the name of the target file downloaded
    :type name: str
    :return: The destination path of the downloaded file
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    if name is None:
        name = os.path.basename(os.path.normpath(url))
    s_file_path = os.path.join(directory, name)
    if not os.path.exists(s_file_path):
        urllib.request.urlretrieve(url, s_file_path)
    else:
        print("File {} already exists and doesn't need to be donwloaded".format(s_file_path))

    return s_file_path

def read_matfile(fname):
    """
    loosely copied on https://stackoverflow.com/questions/29185493/read-svhn-dataset-in-python

    Python function for importing the SVHN data set.
    """
    # Load everything in some numpy arrays
    data = sio.loadmat(fname)
    img = np.moveaxis(data['X'], -1, 0)
    lbl = data['y']
    return img, lbl

def load_svhn_data():
    data_root_url = "http://ufldl.stanford.edu/housenumbers/"
    data_leaf_values = {
            "train": "train_32x32.mat",
            "test": "test_32x32.mat",
    }
    data_arrays = {}

    with tempfile.TemporaryDirectory() as d_tmp:
        for leaf_name, leaf in data_leaf_values.items():
            leaf_url = data_root_url + leaf
            matfile_path = download_data(leaf_url, d_tmp, leaf)
            data_arrays[leaf_name] = read_matfile(matfile_path)

    return data_arrays["train"], data_arrays["test"]

tf.logging.set_verbosity(tf.logging.ERROR)


# Preparing the dataset and parameters#########################


(x_train, y_train), (x_test, y_test) = load_svhn_data()


batch_size = 32
num_classes = 10
epochs = 100
iterations = int(x_train.shape[0] / batch_size)
dropout = 0.5
weight_decay = 1e-4

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
# y - 1 because classes are labeled from 1 to 10 not 0 to 9
y_train = keras.utils.to_categorical(y_train - 1, num_classes)
y_test = keras.utils.to_categorical(y_test - 1, num_classes)

# Build model fromm keras vgg19 implementation
vgg19_model = keras.applications.vgg19.VGG19(include_top=False, weights=None, input_shape=x_train.shape[1:], pooling=None)
model = Sequential()
for layer in vgg19_model.layers:
    model.add(layer)

# Add dense layers on top of convolution
model.add(Flatten(name='flatten'))
model.add(Dense(2048, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(1024, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_classes, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifa10'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(x_train, y_train,
                    batch_size=batch_size),
                    epochs=epochs,
                    steps_per_epoch=iterations,
                    validation_data=(x_test, y_test))

# Save model and weights
save_dir = Path(__file__).parent / "saved_models"
save_dir.mkdir(exist_ok=True, parents=True)
model_name = Path(__file__).stem + "_" + str(int(time.time())) + ".h5"
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('{} Test loss: {}'.format(__file__, scores[0]))
print('{} Test accuracy: {}'.format(__file__, scores[1]))