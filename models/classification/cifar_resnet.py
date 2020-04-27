"""
Inspired from https://github.com/BIGBALLON/cifar-10-cnn/blob/master/4_Residual_Network/ResNet_keras.py
"""
import time

import keras
import argparse
import numpy as np
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D, ZeroPadding2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K

# set GPU memory
if ('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='NUMBER',
                    help='batch size(default: 128)')
parser.add_argument('-e', '--epochs', type=int, default=200, metavar='NUMBER',
                    help='epochs(default: 200)')
# parser.add_argument('-n', '--stack_n', type=int, default=5, metavar='NUMBER',
#                     help='stack number n, total layers = 6 * n + 2 (default: 5)')
parser.add_argument('-d', '--dataset', type=str, default="cifar100", metavar='STRING',
                    help='dataset. (default: cifar10)')
parser.add_argument('-s', '--seed', type=int, metavar='NUMBER',
                    help='Seed number. No default.')
parser.add_argument('-m', '--model', type=str, default="resnet50", metavar='MODEL',
                    help='Model name to use')

args = parser.parse_args()

if args.seed is not None:
    from numpy.random import seed

    seed(args.seed)
    from tensorflow import set_random_seed

    set_random_seed(args.seed)
print(args.seed)

if args.model == "resnet50":
    lst_stack_n = [3, 4, 6, 3]
    lst_o_filter = [256, 512, 1024, 2048]
    lst_strides = [(1, 1), (2, 2), (2, 2), (2, 2)]
    bottleneck = True
elif args.model == "resnet20":
    lst_stack_n = [2, 2, 2, 2]
    lst_o_filter = [64, 128, 256, 512]
    lst_strides = [(1, 1), (2, 2), (2, 2), (2, 2)]
    bottleneck = False
else:
    raise NotImplementedError
# stack_n = args.stack_n
# layers = 6 * stack_n + 2
img_rows, img_cols = 32, 32
img_channels = 3
batch_size = args.batch_size
epochs = args.epochs
iterations = 50000 // batch_size + 1
# weight_decay = 1e-4
weight_decay = 5e-4


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # mean = [125.307, 122.95, 113.865]
    # std = [62.9932, 62.0887, 66.7048]
    # for i in range(3):
    #     x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
    #     x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

    x_train /= 255
    x_test /= 255

    mean_x_train = np.mean(x_train, axis=(0, 1, 2))
    std_x_train = np.std(x_train, axis=(0, 1, 2))
    x_train -= mean_x_train
    x_train /= std_x_train
    x_test -= mean_x_train
    x_test /= std_x_train

    return x_train, x_test


def scheduler(epoch):
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001


def scheduler2(epoch):
    if 0 <= epoch < 60:
        return 0.1
    elif 60 <= epoch < 120:
        return 0.02
    elif 120 <= epoch < 160:
        return 0.004
    return 0.0008


def residual_network(img_input, classes_num, lst_stack_n, lst_o_filter, bottleneck, lst_strides):
    def residual_block(x, o_filters, increase=False, bottleneck=False, stride=(1, 1)):
        # stride = (1, 1)
        # if increase:
        #     stride = (2, 2)

        if bottleneck:
            core_filters = o_filters // 4
            bottle_conv = Conv2D(core_filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                 kernel_initializer="he_normal",
                                 use_bias=False,
                                 kernel_regularizer=regularizers.l2(weight_decay),
                                 bias_regularizer=regularizers.l2(weight_decay))(x)
            bottle_out = Activation('relu')(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_regularizer=regularizers.l2(weight_decay),
                                                               beta_regularizer=regularizers.l2(weight_decay))(bottle_conv))
            bottle_x = bottle_out
        else:
            core_filters = o_filters
            bottle_x = x

        conv_1 = Conv2D(core_filters, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_uniform",
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(weight_decay),
                        bias_regularizer=regularizers.l2(weight_decay))(bottle_x)
        o1 = Activation('relu')(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_regularizer=regularizers.l2(weight_decay),
                                                   beta_regularizer=regularizers.l2(weight_decay))(conv_1))

        conv_2 = Conv2D(o_filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        kernel_initializer="he_uniform",
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(weight_decay),
                        bias_regularizer=regularizers.l2(weight_decay))(o1)
        o2 = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_regularizer=regularizers.l2(weight_decay),
                                beta_regularizer=regularizers.l2(weight_decay))(conv_2)

        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=stride, padding='same',
                                kernel_initializer="he_uniform",
                                use_bias=False,
                                kernel_regularizer=regularizers.l2(weight_decay),
                                bias_regularizer=regularizers.l2(weight_decay))(x)
            projection = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_regularizer=regularizers.l2(weight_decay),
                                            beta_regularizer=regularizers.l2(weight_decay))(projection)
            block = add([o2, projection])
        else:
            block = add([o2, x])
        return block

    # x = ZeroPadding2D(padding=(2,2))(img_input)
    x = img_input
    x = Conv2D(filters=lst_o_filter[0] if not bottleneck else lst_o_filter[0] // 4,
               kernel_size=(3, 3), strides=(1, 1), padding='same',
               use_bias=False,
               kernel_initializer="he_uniform",
               kernel_regularizer=regularizers.l2(weight_decay),
               bias_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5,
                           gamma_regularizer=regularizers.l2(weight_decay),
                           beta_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation("relu")(x)

    assert len(lst_o_filter) == len(lst_stack_n)

    for i in range(len(lst_o_filter)):
        n_o_filter = lst_o_filter[i]
        stack_n = lst_stack_n[i]
        strides = lst_strides[i]

        increase_dim = True
        x = residual_block(x, n_o_filter, increase_dim, bottleneck, stride=strides)

        increase_dim = False
        for _ in range(stack_n - 1):
            x = residual_block(x, n_o_filter, increase_dim, bottleneck, stride=(1, 1))

        # for _ in range(1, stack_n):
        #     x = residual_block(x, 32, False)
        #
        # # input: 16x16x32 output: 8x8x64
        # x = residual_block(x, 64, True)
        # for _ in range(1, stack_n):
        #     x = residual_block(x, 64, False)

    # x = BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    # x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(classes_num,  # activation='softmax',
              kernel_initializer="he_uniform",
              kernel_regularizer=regularizers.l2(weight_decay),
              bias_regularizer=regularizers.l2(weight_decay))(x)
    return x


if __name__ == '__main__':

    print("========================================")
    print(f"MODEL: {args.model}")
    print("BATCH SIZE: {:3d}".format(batch_size))
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))
    print("DATASET: {:}".format(args.dataset))

    print("== LOADING DATA... ==")
    # load data
    global num_classes
    if args.dataset == "cifar100":
        num_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("== DONE! ==\n== COLOR PREPROCESSING... ==")
    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)

    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    img_input = Input(shape=(img_rows, img_cols, img_channels))
    output = residual_network(img_input, num_classes, lst_o_filter=lst_o_filter, lst_stack_n=lst_stack_n, bottleneck=bottleneck,
                              lst_strides=lst_strides)
    resnet = Model(img_input, output)

    # print model architecture if you need.
    print(resnet.summary())

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    categorical_crossentropy = lambda pred, truth: keras.losses.categorical_crossentropy(pred, truth, from_logits=True)
    resnet.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['categorical_accuracy'])

    # set callback
    cbks = [TensorBoard(log_dir='./resnet_{}_{}/'.format(args.model, args.dataset), histogram_freq=0),
            LearningRateScheduler(scheduler2)]

    # dump checkpoint if you need.(add it to cbks)
    # ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)

    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 rotation_range=15,
                                 width_shift_range=4,
                                 height_shift_range=4,
                                 fill_mode='constant', cval=0.,
                                 # featurewise_center=True,
                                 # featurewise_std_normalization=True
                                 )

    datagen.fit(x_train)

    # start training
    resnet.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         validation_data=(x_test, y_test))
    str_seed = "" if args.seed is None else f"_{args.seed}"
    resnet.save('resnet_{}_{}{}_{}.h5'.format(args.model, args.dataset, str_seed, int(time.time())))
