import tensorflow as tf

# Parameters
channels = 1

def summary_layer(net, name):
    tf.summary.image(name, tf.expand_dims(tf.transpose(net, [3, 0, 1, 2])[0], 3), max_outputs=1)
    print(net.shape)


def autoencoder(inputs, batch_size, dropout=0.5):
    # Encoder
    net = tf.layers.conv2d(inputs, 128, [4, 4], strides=(3, 3), padding='SAME')
    summary_layer(net, 'conv1')

    tf.layers.dropout(net, dropout)
    net = tf.layers.conv2d(net, 64, [3, 3], strides=(2, 2), padding='SAME')
    summary_layer(net, 'conv2')

    # Decoder
    tf.layers.dropout(net, dropout)
    net = tf.layers.conv2d_transpose(net, 64, [3, 3], strides=(2, 2), padding='SAME')
    summary_layer(net, 'deconv1')

    tf.layers.dropout(net, dropout)
    net = tf.layers.conv2d_transpose(net, channels, [4, 4], strides=(3, 3), padding='SAME')
    summary_layer(net, 'deconv2')

    # Outlier rejection (no dropout)
    #net = tf.layers.conv2d(net, channels, [1, 1], strides=(1, 1), padding='SAME')
    #summary_layer(net, 'outlier_rej')

    # Final tanh activation
    net = tf.nn.tanh(net)
    return net
