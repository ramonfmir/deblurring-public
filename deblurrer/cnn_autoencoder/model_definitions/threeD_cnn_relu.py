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

    filter_1 = tf.Variable(tf.truncated_normal([10, 30, channels, 10], stddev=0.05))
    net = tf.nn.conv2d(inputs, filter_1, strides=[1, 1, 1, 1], padding='VALID')
    net = tf.nn.relu(net)
    print(net.shape)

    net = tf.expand_dims(net,1)
    net = tf.transpose(net, [0, 4, 2, 3, 1])

    print(net.shape)

    filter_2 = tf.Variable(tf.truncated_normal([10, 10, 30, 1, 1], stddev=0.05))
    net = tf.nn.conv3d(net, filter_2, strides=[1, 1, 1, 1, 1], padding='VALID')
    net = tf.nn.relu(net)
    print(net.shape)

    filter_3 = tf.Variable(tf.truncated_normal([1, 19, 59, 1, 1], stddev=0.05))
    net = tf.nn.conv3d_transpose(net, filter_3, strides=[1, 1, 1, 1, 1], output_shape=[batch_size, 1, 90, 270, 1], padding='VALID')
    net = tf.nn.relu(net)
    print('final net: ', net.shape)

    net = tf.squeeze(net, axis=1)
    print(net)

    return net
