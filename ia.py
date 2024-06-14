import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
import numpy


def build_model(conv_size, conv_depth):
    board3d = layers.Input(shape=(14, 8, 8))

    x = board3d
    for _ in range(conv_depth):
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu',
                          )(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, 'relu')(x)
    x = layers.Dense(1, 'sigmoid')(x)

    return models.Model(inputs=board3d, outputs=x)


def train():
    model = build_model(32, 4)

    def get_dataset():
        container = numpy.load('dataset.npz')
        b, v = container['b'], container['v']
        v = numpy.asarray(v / abs(v).max() / 2 + 0.5)  # normalization (0 - 1)
        return b, v

    x_train, y_train = get_dataset()

    print(x_train.shape)
    print(y_train.shape)

    model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')

    model.summary()

    model.fit(x=x_train,
              y=y_train,
              batch_size=2048,
              epochs=10,
              verbose=1,
              validation_split=0.1,
              callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                         callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4)])

    model.save('model.h5')


def open_model():
    model = models.load_model('model.h5')
    return model



