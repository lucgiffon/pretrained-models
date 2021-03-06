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
import keras
import scipy.io as sio

from keras.callbacks import LearningRateScheduler, TensorBoard
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from conv_vgg19 import VGG19


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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', type=int, metavar='NUMBER',
                    help='Seed number. No default.')
parser.add_argument('-d', '--size-dense', type=int, metavar='NUMBER', default=2048,
                    help='Seed number. No default.')

args = parser.parse_args()

if args.seed is not None:
    from numpy.random import seed
    seed(args.seed)
    from tensorflow import set_random_seed
    set_random_seed(args.seed)

SIZE_DENSE = args.size_dense
print(args.seed, SIZE_DENSE)
def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 160:
        return 0.01
    return 0.001

# Preparing the dataset and parameters#########################


(x_train, y_train), (x_test, y_test) = load_svhn_data()


batch_size = 128
num_classes = 10
epochs = 300
iterations = int(x_train.shape[0] / batch_size)
dropout = 0.5
weight_decay = 1e-4

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
mean_x_train = np.mean(x_train, axis=(0, 1, 2))
std_x_train = np.std(x_train, axis=(0, 1, 2))
x_train -= mean_x_train
x_train /= std_x_train
mean_x_test = np.mean(x_test, axis=(0, 1, 2))
std_x_test = np.std(x_test, axis=(0, 1, 2))
x_test -= mean_x_test
x_test /= std_x_test

# Convert class vectors to binary class matrices.
# y - 1 because classes are labeled from 1 to 10 not 0 to 9
y_train = keras.utils.to_categorical(y_train - 1, num_classes)
y_test = keras.utils.to_categorical(y_test - 1, num_classes)

str_seed = "" if args.seed is None else f"_{args.seed}"
model_base_name = Path(__file__).stem + "_" + "{}x{}".format(SIZE_DENSE, SIZE_DENSE) + str_seed + "_" + str(int(time.time()))# important note: we do not use vgg19 from keras application because training from scratch for the cifar10 task doesn't provide satisfying results.
# Because it doesn't contain batchnorm layer nor kernel regularizers in convolution blocks. This is why we use this custom VGG19 function.
# vgg19_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=x_train.shape[1:], pooling=None)
model = VGG19(size_denses=SIZE_DENSE, input_shape=x_train.shape[1:], num_classes=num_classes, dropout=dropout, weight_decay=weight_decay)

# Add dense layers on top of convolution

# initiate RMSprop optimizer
opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
# opt = keras.optimizers.RMSprop(lr=0.0001)
# opt = keras.optimizers.Adam(lr=0.001)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:

datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant',
                             cval=0.)

datagen.fit(x_train)

change_lr = LearningRateScheduler(scheduler)

# disabled early stopping because it somehow lead to worse results
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
#                               min_delta=0,
#                               patience=3,
#                               verbose=0, mode='auto')

tb_cb = TensorBoard(log_dir="tb_{}".format(model_base_name), histogram_freq=0)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(x_train, y_train,
                    batch_size=batch_size),
                    epochs=epochs,
                    steps_per_epoch=iterations,
                    validation_data=(x_test, y_test),
                    callbacks=[change_lr, tb_cb]
                    )

# Save model and weights
save_dir = Path(__file__).parent / "saved_models"
save_dir.mkdir(exist_ok=True, parents=True)
model_name = model_base_name + ".h5"
model_path = save_dir / model_name
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('{} Test loss: {}'.format(__file__, scores[0]))
print('{} Test accuracy: {}'.format(__file__, scores[1]))
