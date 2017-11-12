import tensorflow as tf

# Parameters
channels = 1

def summary_layer(net, name):
    tf.summary.image(name, tf.expand_dims(tf.transpose(net, [3, 0, 1, 2])[0], 3), max_outputs=1)
    print(net.shape)

def autoencoder(inputs, batch_size, dropout=0.65):
    # Encoder
    net = tf.layers.conv2d(inputs, 64, [5, 5], strides=(3, 3), padding='SAME')
    summary_layer(net, 'conv1')

    net = tf.layers.dropout(net, dropout)
    net = tf.layers.conv2d(net, 64, [5, 5], strides=(2, 2), padding='SAME')
    summary_layer(net, 'conv2')

    # Decoder
    net = tf.layers.dropout(net, dropout)
    net = tf.layers.conv2d_transpose(net, 64, [5, 5], strides=(2, 2), padding='SAME')
    summary_layer(net, 'deconv1')

    net = tf.layers.dropout(net, dropout)
    net = tf.layers.conv2d_transpose(net, channels, [4, 4], strides=(3, 3), padding='SAME')
    summary_layer(net, 'deconv2')

    # Final tanh activation
    net = tf.nn.tanh(net)
    return net
