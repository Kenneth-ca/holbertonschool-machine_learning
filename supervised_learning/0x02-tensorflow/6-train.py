#!/usr/bin/env python3
"""
Module that builds, trains and saves a neural network classifier
"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    a function builds, trains and saves a neural network classifier
    :param X_train: a numpy.ndarray containing the training input data
    :param Y_train: a numpy.ndarray containing the training labels
    :param X_valid: a numpy.ndarray containing the training input data
    :param Y_valid: a numpy.ndarray containing the training labels
    :param layer_sizes: list containing the number of nodes in each layer
    :param activations: list containing the activation functions of each layer
    :param alpha: the learning rate
    :param iterations: the number of iterations to train over
    :param save_path: designates where to save the model
    :return: an operation that trains the network using gradient descent
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection("y_pred", y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)

    training = create_train_op(loss, alpha)
    tf.add_to_collection("train", train)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            # cost and accuracy for training and validation sets
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = sess.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(accuracy_train))
                print("\tValidation Cost: {}".format(cost_val))
                print("\tValidation Accuracy: {}".format(accuracy_val))
            # training for each iteration
            if i < iterations:
                sess.run(training, feed_dict={x: X_train, y: Y_train})
        return saver.save(sess, save_path)
