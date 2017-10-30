import tensorflow.contrib.layers as lays
import tensorflow as tf

# Parameters
channels = 1

def autoencoder(inputs, batch_size):
    # Encoder
    net = lays.conv2d(inputs, 32, [5, 5], stride=3, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=1, padding='SAME')

    # Decoder
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=1, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, channels, [5, 5], stride=3, padding='SAME', activation_fn=tf.nn.tanh)

    return net
