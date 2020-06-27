#!/usr/bin/env python3
"""
Trains a convolutional neural network to classify the CIFAR 10 dataset:
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    a function that trains a convolutional neural network to classify the
    CIFAR 10 dataset
    :param X: X is a numpy.ndarray of shape (m, 32, 32, 3) containing the
    CIFAR 10 data, where m is the number of data points
    :param Y: Y is a numpy.ndarray of shape (m,) containing the CIFAR 10
    labels for X
    :return: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    (x_train, y_train), _ = K.datasets.cifar10.load_data()
    print((x_train.shape, y_train.shape))
    x_train, y_train = preprocess_data(x_train, y_train)
    print((x_train.shape, y_train.shape))

    input_tensor = K.Input(shape=(32, 32, 3))
    x = K.layers.UpSampling2D((2, 2))(input_tensor)
    x = K.layers.UpSampling2D((2, 2))(x)
    x = K.layers.UpSampling2D((2, 2))(x)
    model = K.applications.ResNet50(include_top=False,
                                    weights="imagenet",
                                    input_tensor=x)

    last_layer = model.layers[-1].output

    FC = K.layers.Flatten()(last_layer)
    FC = K.layers.BatchNormalization()(FC)
    FC = K.layers.Dense(128, activation='relu')(FC)
    FC = K.layers.Dropout(0.5)(FC)
    FC = K.layers.BatchNormalization()(FC)
    FC = K.layers.Dense(64, activation='relu')(FC)
    FC = K.layers.Dropout(0.5)(FC)
    FC = K.layers.BatchNormalization()(FC)
    FC = K.layers.Dense(units=10, activation="softmax")(FC)

    model = K.models.Model(inputs=model.input, outputs=FC)

    check_point = K.callbacks.ModelCheckpoint(filepath="cifar10.h5",
                                              monitor="acc",
                                              mode="max",
                                              save_best_only=True,
                                              )

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1,
                        callbacks=[check_point])
    model.save("cifar10.h5")
