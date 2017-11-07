import tensorflow.contrib.layers as lays
import tensorflow as tf

# Parameters
channels = 1

def autoencoder(inputs, batch_size):
    # Encoder
    net = lays.conv2d(inputs, 256, [5, 5], stride=3, padding='SAME')
    print(net.shape)
    net = lays.conv2d(net, 128, [5, 5], stride=2, padding='SAME')
    print(net.shape)
    net = lays.conv2d(net, 64, [5, 5], stride=1, padding='SAME')
    print(net.shape)

    # Decoder
    net = lays.conv2d_transpose(net, 128, [5, 5], stride=1, padding='SAME')
    print(net.shape)
    net = lays.conv2d_transpose(net, 256, [5, 5], stride=2, padding='SAME')
    print(net.shape)
    net = lays.conv2d_transpose(net, channels, [5, 5], stride=3, padding='SAME', activation_fn=tf.nn.tanh)
    print(net.shape)

    return net
