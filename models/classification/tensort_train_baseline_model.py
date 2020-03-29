from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense
from keras import regularizers
from keras.models import Sequential


def tensor_train_baseline(input_shape, num_classes, weight_decay=1e-4):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same'))

    model.add(Flatten())
    model.add(Dense(1536,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # model.add(TTLayerConv(window=[3, 3], inp_modes=[4, 4, 4, 8], out_modes=[4, 4, 4, 8], mat_ranks=self.tt_rank_conv))
    # model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(TTLayerConv(window=[3, 3], inp_modes=[4, 4, 4, 8], out_modes=[4, 4, 4, 8], mat_ranks=self.tt_rank_conv))
    # model.add(TTLayerDense(inp_modes=[4, 4, 4, 8], out_modes=[4, 4, 4, 8], mat_ranks=self.tt_rank))
    # model.add(TTLayerDense(inp_modes=[4, 4, 4, 8], out_modes=[10, 1, 1, 1], mat_ranks=self.tt_rank))
    return model