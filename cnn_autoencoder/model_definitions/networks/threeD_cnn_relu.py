import tensorflow.contrib.layers as lays
import tensorflow as tf
import numpy as np

# Parameters
channels = 1
depth = 4

def autoencoder(inputs, batch_size):
    # encoder
    print(inputs.shape)
    inputs = tf.expand_dims(inputs, 1)

    print(inputs.shape)

    filter_1 = tf.Variable(tf.truncated_normal([1, 10, 30, channels, depth], stddev=0.05))
    net = tf.nn.conv3d(inputs, filter_1, strides=[1, 1, 3, 3, 1], padding='SAME')
    net = tf.nn.tanh(net)
    print(net.shape)

    # tf.summary.image("Intermediate", tf.transpose(tf.expand_dims(tf.transpose(net, [3, 1, 2, 0])[0], 0), [3, 1, 2, 0]), max_outputs=2)

    # net = tf.transpose(net, [0, 4, 2, 3, 1])

    filter_3 = tf.Variable(tf.truncated_normal([1, 10, 30, 1, depth], stddev=0.05))
    net = tf.nn.conv3d_transpose(net, filter_3, strides=[1, 1, 3, 3, 1], output_shape=[batch_size, 1, 90, 270, 1], padding='SAME')
    net = tf.nn.tanh(net)
    print('final net: ', net.shape)

    net = tf.squeeze(net, axis=1)
    print(net)

    return net

# def autoencoder(inputs, batch_size):
#     # encoder
#     print(inputs.shape)
#
#     filter_1 = tf.Variable(tf.truncated_normal([10, 30, channels, depth], stddev=0.05))
#     net = tf.nn.conv2d(inputs, filter_1, strides=[1, 1, 1, 1], padding='VALID')
#     net = tf.nn.relu(net)
#     print(net.shape)
#
#     tf.summary.image("Intermediate", tf.transpose(tf.expand_dims(tf.transpose(net, [3, 1, 2, 0])[0], 0), [3, 1, 2, 0]), max_outputs=2)
#
#     net = tf.expand_dims(net, 1)
#     # net = tf.transpose(net, [0, 4, 2, 3, 1])
#
#     print(net.shape)
#
#     filter_2 = tf.Variable(tf.truncated_normal([1, 10, 30, depth, depth], stddev=0.05))
#     net = tf.nn.conv3d(net, filter_2, strides=[1, 1, 1, 1, 1], padding='SAME')
#     net = tf.nn.relu(net)
#     print(net.shape)
#
#     filter_3 = tf.Variable(tf.truncated_normal([1, 10, 30, 1, depth], stddev=0.05))
#     net = tf.nn.conv3d_transpose(net, filter_3, strides=[1, 1, 1, 1, 1], output_shape=[batch_size, 1, 90, 270, 1], padding='VALID')
#     net = tf.nn.tanh(net)
#     print('final net: ', net.shape)
#
#     net = tf.squeeze(net, axis=1)
#     print(net)
#
#     return net
