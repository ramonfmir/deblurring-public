import tensorflow.contrib.layers as lays
import tensorflow as tf

# Parameters
channels = 1

def autoencoder(inputs):
    # Deconvolution Sub-Network
    net = lays.conv2d(inputs, 38, [1, 181], stride=1)
    net = lays.conv2d(net, 1, [61, 1], stride=1)
    net = lays.conv3d(net, 512, [16, 16, 512], stride=1)
    # Outlier Rejection Sub-Network
    net = lays.conv3d(net, 512, [1, 1, 512], stride=1)
    net = lays.conv3d(net, 1, [8, 8, 512], stride=1)
    # Resoration
    return net
