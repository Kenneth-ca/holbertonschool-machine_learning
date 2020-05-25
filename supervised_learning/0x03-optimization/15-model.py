#!/usr/bin/env python3
"""
Optimizes a neural network with tensorflow
"""
import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    """
    a function that create placeholders
    :param nx: the number of feature columns in our data
    :param classes: the number of classes in our classifier
    :return: placeholders named x and y, respectively
    """
    return tf.placeholder(float, shape=[None, nx], name='x'), tf.placeholder(
        float, shape=[None, classes], name='y')


def create_layer(prev, n, activation):
    """
    a function that create layers
    :param prev: the tensor output of the previous layer
    :param n: the number of nodes in the layer to create
    :param activation: is the activation function that the layer should use
    :return: the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            name="layer")
    return layer(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    a function that creates the forward propagation graph
    :param x: the placeholder for the input data
    :param layer_sizes: a list contating the number of nodes in each layer
    :param activations: a list containing the activation functions
    :return: the prediction of the network in tensor form
    """
    prediction = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for layer in range(1, len(layer_sizes)):
        if layer != len(layer_sizes) - 1:
            prediction = create_batch_norm_layer(prediction, layer_sizes[
                layer], activations[layer])
        else:
            prediction = create_layer(prediction, layer_sizes[layer],
                                      activations[layer])
    return prediction


def calculate_accuracy(y, y_pred):
    """
    a function that calculates the accuracy of a prediction
    :param y: a placeholders with the right labels of the input data
    :param y_pred: tensor containing the network's predictions
    :return: a tensor containing the decimal accuracy of the prediction
    """
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    mean = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    return mean


def calculate_loss(y, y_pred):
    """
    a function that calculates the loss of a prediction
    :param y: a placeholders with the right labels of the input data
    :param y_pred: tensor containing the network's predictions
    :return: a tensor containing the loss of a prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def shuffle_data(X, Y):
    """
    a function that shuffles the data points in two matrices the same way
    :param X: the first numpy.ndarray of shape (m, nx) to shuffle
    :param Y: the second numpy.ndarray of shape (m, ny) to shuffle
    :return: the shuffled X and Y matrices
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation, :], Y[permutation, :]


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    a function that optimizes using Adam with tensorflow
    :param alpha: the learning rate
    :param beta1: the weight used for the first moment
    :param beta2: the weight used for the second moment
    :param epsilon: small number to avoid division by zero
    :param var: numpy.ndarray containing the variable to be updated
    :param grad: numpy.ndarray containing the gradient of var
    :param v: the previous first moment of var
    :param s: the previous second moment of var
    :param t: the time step used for bias correction
    :return: the updated variable, the new first moment, and the new second
    moment, respectively
    """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    a function that updates the learning rate with tensorflow
    :param alpha: the original learning rate
    :param decay_rate: the weight used to determine the rate at which alpha
    will decay
    :param global_step: the number of passes of gradient descent that have
    elapsed
    :param decay_step: the number of passes of gradient descent that should
    occur before alpha is decayed further
    :return: the learning rate decay operation
    """
    # staircase= True will make discrete output as floor(global_step /
    # decay_step in the calculation)
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)


def create_batch_norm_layer(prev, n, activation):
    """
    a function that uses batch normalization with tensorflow
    :param prev: the activated output of the previous layer
    :param n: the number of nodes in the layer to be created
    :param activation: the activation function that should be used on the
    output of the layer
    :return: a tensor of the activated output for the layer
    """
    # Layers
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output = tf.layers.Dense(units=n, kernel_initializer=k_init)
    Z = output(prev)

    # Gamma and Beta initialization
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name="gamma")
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]), name="beta")

    # Batch normalization
    mean, var = tf.nn.moments(Z, axes=0)
    b_norm = tf.nn.batch_normalization(Z, mean, var, offset=beta,
                                       scale=gamma,
                                       variance_epsilon=1e-8)
    if activation is None:
        return b_norm
    else:
        return activation(b_norm)


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """
    a function that optimizes a neural network model with tensorflow
    :param Data_train: tuple containing the training inputs and training
    labels, respectively
    :param Data_valid: tuple containing the validation inputs and validation
    labels, respectively
    :param layers: a list containing the number of nodes in each layer
    :param activations: a list containing the activation functions used for
    each layer of the network
    :param alpha: the learning rate
    :param beta1: the weight for the first moment of Adam Optimization
    :param beta2: the weight for the second moment of Adam Optimization
    :param epsilon: a small number used to avoid division by zero
    :param decay_rate: the decay rate for inverse time decay of the learning
    rate (the corresponding decay step should be 1)
    :param batch_size: the number of data points that should be in a mini-batch
    :param epochs: the number of times the training should pass through the
    whole dataset
    :param save_path: the path where the model should be saved to
    :return: the path where the model was saved
    """
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid

    x, y = create_placeholders(nx, classes)
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection("y_pred", y_pred)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)

    global_step = tf.Variable(0)
    alpha_d = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha_d, beta1, beta2, epsilon)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        m = X_train.shape[0]
        # mini batch definition
        if m % batch_size == 0:
            n_batches = m // batch_size
        else:
            n_batches = m // batch_size + 1

        # training loop
        for i in range(epochs + 1):
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = sess.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accuracy_val))

            if i < epochs:
                shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

                # mini batches
                for b in range(n_batches):
                    start = b * batch_size
                    end = (b + 1) * batch_size
                    if end > m:
                        end = m
                    X_mini_batch = shuffled_X[start:end]
                    Y_mini_batch = shuffled_Y[start:end]

                    next_train = {x: X_mini_batch, y: Y_mini_batch}
                    sess.run(train_op, feed_dict=next_train)

                    if (b + 1) % 100 == 0 and b != 0:
                        loss_mini_batch = sess.run(loss, feed_dict=next_train)
                        acc_mini_batch = sess.run(accuracy,
                                                  feed_dict=next_train)
                        print("\tStep {}:".format(b + 1))
                        print("\t\tCost: {}".format(loss_mini_batch))
                        print("\t\tAccuracy: {}".format(acc_mini_batch))

            # Update of global step variable for each iteration
            sess.run(tf.assign(global_step, global_step + 1))

        return saver.save(sess, save_path)
