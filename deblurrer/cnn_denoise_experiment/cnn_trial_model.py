import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
from skimage import transform
import model

# Parameters
channels = 1
corruption_level = 0.2

def autoencoder(inputs):
    # encoder
    # 270 x 90 x 1   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  8 x 8 x 16
    # 8 x 8 x 16    ->  2 x 2 x 8
    net = lays.conv2d(inputs, 32, [5, 5], stride=3, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=1, padding='SAME')
    # decoder
    # 2 x 2 x 8    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=1, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, channels, [5, 5], stride=3, padding='SAME', activation_fn=tf.nn.relu)
    return net

def initialise(image_width, image_height, lr=0.01):
    original = tf.placeholder(tf.float32, (None, image_height, image_width, 3))  # input to the network (MNIST images)
    original_greyscale = tf.reduce_mean(original, axis=3, keep_dims = True)

    corrupted = tf.placeholder(tf.float32, (None, image_height, image_width, 3))  # input to the network (MNIST images)
    corrupted_greyscale = tf.reduce_mean(corrupted, axis=3,keep_dims = True)

    deblurred = autoencoder(corrupted)  # create the Autoencoder network

    # calculate the loss and optimize the network
    cost = tf.reduce_mean(tf.square(deblurred - original_greyscale))  # claculate the mean square error loss
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # initialize the network
    init = tf.global_variables_initializer()

    return model.Model(train_op, cost, original, corrupted, deblurred, init)
