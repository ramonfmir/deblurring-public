import tensorflow.contrib.layers as lays
import tensorflow as tf
import numpy as np

# Parameters
channels = 1

# def autoencoder(inputs):
#     # Deconvolution Sub-Network
#     net = lays.conv2d(inputs, 38, [1, 181], stride=1)
#     net = lays.conv2d(net, 1, [61, 1], stride=1)
#     net = lays.conv3d(net, 512, [16, 16, 512], stride=1)
#     # Outlier Rejection Sub-Network
#     net = lays.conv3d(net, 512, [1, 1, 512], stride=1)
#     net = lays.conv3d(net, 1, [8, 8, 512], stride=1)
#     # Resoration
#     return net

def autoencoder(inputs, batch_size):
    # encoder
    print(inputs.shape)

    filter_1 = tf.Variable(tf.truncated_normal([1,181, channels, 38], stddev=0.05))
    net = tf.nn.conv2d(inputs, filter_1, strides=[1, 1, 1, 1], padding='VALID')
    net = tf.nn.relu(net)
    print(net.shape)

    filter_2 = tf.Variable(tf.truncated_normal([61, 1, 38, 38], stddev=0.05))
    net = tf.nn.conv2d(net, filter_2, strides=[1, 1, 1, 1], padding='VALID')
    net = tf.nn.relu(net)
    print(net.shape)

    net = tf.expand_dims(net,4)
    print(net.shape)
    filter_3 = tf.Variable(tf.truncated_normal([16, 16,38, 1, 10], stddev=0.05))
    net = tf.nn.conv3d(net, filter_3, strides=[1, 1, 1, 1, 1], padding='VALID')
    net = tf.nn.relu(net)
    print(net.shape)

    # filter_4 = tf.Variable(tf.truncated_normal([5, 5, 1, 512, 512], stddev=0.05))
    # net = tf.nn.conv3d(net, filter_4, strides=[1, 1, 1, 1, 1], padding='VALID')
    # net = tf.nn.relu(net)
    # print(net.shape)
    #
    # filter_5 = tf.Variable(tf.truncated_normal([5, 5, 1, 512, 512], stddev=0.05))
    # net = tf.nn.conv3d(net, filter_5, strides=[1, 1, 1, 1, 1], padding='VALID')
    # net = tf.nn.relu(net)
    # print(net.shape)

    net = tf.squeeze(net, [3])
    print(net.shape)
    # decoder
    filter_t1 = tf.Variable(tf.truncated_normal([76, 196, channels, 10], stddev=0.05))
    net = tf.nn.conv2d_transpose(net, filter_t1, output_shape=[batch_size, 90, 270, 1], strides=[1, 1, 1, 1], padding='VALID')
    net = tf.nn.relu(net)

    print(net.shape)
    return net
