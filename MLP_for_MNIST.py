"""
Work in Progress
"""

import numpy as np
import tensorflow as tf


def relu(x):
    return np.maximum(0, x)


def derivative_relu(x):
    temp = np.zeros(x)
    temp[x >= 0] = 1
    return temp


def sigmoid(x):
    x = x - np.max(x)
    return 1.0 / (1.0 + np.exp(x))


def derivative_sigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return exps / np.expand_dims(np.sum(exps, axis=-1), axis=-1)


class NeuralNetwork:
    def __init__(self, num_input, num_hidden, num_output, lr):
        # num of neurons in layers
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        # learning rate
        self.lr = lr

        # initialise
        self.w_ih = np.random.normal(0.0, 1.0, (self.num_hidden, self.num_input)) / np.sqrt(self.num_input)
        self.w_ho = np.random.normal(0.0, 1.0, (self.num_output, self.num_hidden)) / np.sqrt(self.num_hidden)

    def feedforward(self, _input):
        # hidden layer
        x_h = np.dot(_input, self.w_ih.T)
        o_h = sigmoid(x_h)

        # output layer
        x_o = np.dot(o_h, self.w_ho.T)
        o_o = softmax(x_o)
        return o_o, o_h

    def backprop(self, _input, label):
        o_o, o_h = self.feedforward(_input)

        # calculate error
        e_o = label - o_o
        e_h = np.dot(e_o, self.w_ho)

        # output -> hidden layer
        # Note: derivative of softmax is just e_o
        # w_ho = w_ho - lr * derivative_softmax @ output_hidden
        l = self.lr * np.dot(e_o.T, o_h)
        self.w_ho -= self.lr * np.dot(e_o.T, o_h)

        # hidden layer -> input layer
        # Note: derivative of sigmoid
        self.w_ih -= self.lr * np.dot(derivative_sigmoid(o_h).T, _input)


if __name__ == '__main__':
    # hyperparameters
    num_input = 784
    num_hidden = 100
    num_output = num_class = 10
    num_minibatch = 10000
    num_epoch = 10
    lr = 0.01

    # data preprocessing
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] ** 2)  # flatten the array
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] ** 2)  # flatten the array
    y_train = np.eye(num_class)[y_train]  # manual one hot encoding
    x_train = x_train / 255.0  # normalise the images
    x_test = x_test / 255.0  # normalise the images
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # instantiate the NN
    nn = NeuralNetwork(num_input, num_hidden, num_output, lr)

    # Learning phase
    for e in range(num_epoch):
        # print(np.mean(nn.w_ih), np.mean(nn.w_ho))
        minibatch_index = np.random.randint(low=0, high=x_train.shape[0], size=num_minibatch)
        nn.backprop(x_train[minibatch_index, :], y_train[minibatch_index, :])

    # Evaluation phase
    predict, _ = nn.feedforward(x_test)
    predict = np.argmax(predict, axis=-1)
    score = np.average(predict == y_test)
    print("Performance: {}".format(score))
