# !/usr/bin/env python3
"""
Trains a convolutional neural network to classify the CIFAR 10 dataset:
"""
import tensorflow.keras as K
import tensorflow as tf


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
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    print((x_train.shape, y_train.shape))
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    print((x_train.shape, y_train.shape))

    input_t = K.Input(shape=(32, 32, 3))
    res_model = K.applications.ResNet50(include_top=False,
                                        weights="imagenet",
                                        input_tensor=input_t)

    for layer in res_model.layers[:143]:
        layer.trainable = False
    # Check the freezed was done ok
    for i, layer in enumerate(res_model.layers):
        print(i, layer.name, "-", layer.trainable)

    to_res = (224, 224)

    model = K.models.Sequential()
    model.add(K.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
    model.add(res_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(256, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(10, activation='softmax'))

    check_point = K.callbacks.ModelCheckpoint(filepath="cifar10.h5",
                                              monitor="val_acc",
                                              mode="max",
                                              save_best_only=True,
                                              )

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[check_point])
    model.summary()
    model.save("cifar10.h5")
