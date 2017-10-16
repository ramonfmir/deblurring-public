import tensorflow as tf
import math
import numpy as np

# Parameters
mnist_width = 28
layers = []

# Hyperparameters
alpha = 0.005

# Input placeholders for original image and its corrupted version
original = tf.placeholder("float", [None, 784])
corrupted = tf.placeholder("float", [None, 784])
x = corrupted

# def build_layers(x):
#     shapes = []
#     encoder = []
#     filter_size = 3
#     n_input = 1
#     n_output = 10
#     x = tf.reshape(x, shape=[-1, 28, 28, 1])
#     W = tf.Variable(
#         tf.random_uniform([
#             filter_size,
#             filter_size,
#             n_input, n_output],
#             -1.0 / math.sqrt(n_input),
#             1.0 / math.sqrt(n_input)))
#     encoder.append(W)
#     conv1 = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
#     shapes.append(x.shape.as_list())
#
#     shapes.reverse()
#     encoder.reverse()
#
#     deconv1 = tf.nn.conv2d_transpose(
#                 conv1, encoder[0],
#                 tf.stack([tf.shape(original)[0], shapes[0][1], shapes[0][2], shapes[0][3]]),
#                 strides=[1, 2, 2, 1], padding='SAME')
#     return deconv1

def autoencoder(input_shape=[None, 784],
                n_filters=[1, 5],
                filter_sizes=[2, 2],
                corruption=False):

    # Converted to square tensor
    x_dim = np.sqrt(x.get_shape().as_list()[1])
    x_dim = int(x_dim)
    current_input = tf.reshape(
        x, [-1, x_dim, x_dim, n_filters[0]])

    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        current_input = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')

    # store the latent representation
    encoder.reverse()
    shapes.reverse()

    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        current_input = tf.nn.conv2d_transpose(
                        current_input, W,
                        tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                        strides=[1, 2, 2, 1], padding='SAME')

    return current_input

# Build the CNN
deblurred = autoencoder()

# Cost is defined as error between original and reproduced
reshaped_orig = tf.reshape(original, shape=[-1, 28, 28, 1])
cost = tf.reduce_sum(tf.square(deblurred - reshaped_orig))  # minimize squared error
train_op = tf.train.AdamOptimizer(alpha).minimize(cost)  # construct an optimizer

init = tf.global_variables_initializer()
