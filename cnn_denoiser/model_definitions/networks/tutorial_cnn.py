import tensorflow.contrib.layers as lays
import tensorflow as tf

# Parameters
channels = 1

def autoencoder(inputs, batch_size):
    # Encoder
    net = lays.conv2d(inputs, 128, [5, 5], stride=3, padding='SAME')
    tf.summary.image('conv1', tf.expand_dims(tf.transpose(net, [3, 0, 1, 2])[0], 3), max_outputs=1)
    print(net.shape)
    net = lays.conv2d(net, 64, [5, 5], stride=2, padding='SAME')
    tf.summary.image('conv2', tf.expand_dims(tf.transpose(net, [3, 0, 1, 2])[0], 3), max_outputs=1)
    print(net.shape)
    net = lays.conv2d(net, 64, [5, 5], stride=1, padding='SAME')
    print(net.shape)
    tf.summary.image('code_layer', tf.expand_dims(tf.transpose(net, [3, 0, 1, 2])[0], 3), max_outputs=1)

    # Decoder
    net = lays.conv2d_transpose(net, 64, [5, 5], stride=1, padding='SAME')
    tf.summary.image('deconv1', tf.expand_dims(tf.transpose(net, [3, 0, 1, 2])[0], 3), max_outputs=1)
    print(net.shape)
    # net = lays.conv2d_transpose(net, 256, [5, 5], stride=2, padding='SAME')
    # tf.summary.image('deconv2', tf.expand_dims(tf.transpose(net, [3, 0, 1, 2])[0], 3), max_outputs=1)
    # print(net.shape)
    net = lays.conv2d_transpose(net, channels, [6, 6], stride=6, padding='SAME', activation_fn=tf.nn.tanh)
    print(net.shape)

    return net
