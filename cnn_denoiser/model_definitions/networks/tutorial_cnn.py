import tensorflow as tf

# Parameters
channels = 1

def summary_layer(net, name):
    tf.summary.image(name, tf.expand_dims(tf.transpose(net, [3, 0, 1, 2])[0], 3), max_outputs=1)
    print(net.shape)


def autoencoder(inputs, batch_size, dropout=0.65):
    inits = tf.constant(0.1)
    # Encoder
    pre_lay1 = tf.layers.conv2d(inputs, 32, [3, 3], strides=(1, 1), padding='SAME')

    lay1 = tf.layers.conv2d(pre_lay1, 32, [3, 3], strides=(1, 1), padding='SAME')
    k1 = tf.Variable(inits)
    lay1_ = k1 * lay1 + pre_lay1
    summary_layer(lay1, 'conv1')
    summary_layer(lay1_, 'conv1_')

    # net = tf.layers.dropout(net, dropout)
    lay2 = tf.layers.conv2d(lay1_, 32, [5, 5], strides=(1, 1), padding='SAME')
    k2 = tf.Variable(inits)
    lay2_ = k2 * lay2 + lay1_
    summary_layer(lay2, 'conv2')
    summary_layer(lay2_, 'conv2_')

    # net = tf.layers.dropout(net, dropout)
    lay3 = tf.layers.conv2d(lay2_, 32, [3, 3], strides=(1, 1), padding='SAME')
    k3 = tf.Variable(inits)
    lay3_ = k3 * lay3 + lay2_
    summary_layer(lay3, 'conv3')
    summary_layer(lay3_, 'conv3_')

    # net = tf.layers.dropout(net, dropout)
    lay4 = tf.layers.conv2d(lay3_, 32, [3, 3], strides=(1, 1), padding='SAME')
    k4 = tf.Variable(inits)
    lay4_ = k4 * lay4 + lay3_
    summary_layer(lay3, 'conv4')
    summary_layer(lay3_, 'conv4_')

    # net = tf.layers.dropout(net, dropout)
    net = tf.layers.conv2d(lay4_, 1, [3, 3], strides=(1, 1), padding='SAME')
    summary_layer(net, 'conv5')


    # net = tf.layers.dropout(net, dropout)
    # net = tf.layers.conv2d(net, 64, [3, 3], strides=(2, 2), padding='SAME')
    # summary_layer(net, 'conv3')
    #
    # # Decoder
    # net = tf.layers.dropout(net, dropout)
    # net = tf.layers.conv2d_transpose(net, 64, [3, 3], strides=(2, 2), padding='SAME')
    # summary_layer(net, 'deconv1')
    #
    # net = tf.layers.dropout(net, dropout)
    # net = tf.layers.conv2d_transpose(net, channels, [4, 4], strides=(3, 3), padding='SAME')
    # summary_layer(net, 'deconv2')

    # Outlier rejection (no dropout)
    #net = tf.layers.conv2d(net, channels, [1, 1], strides=(1, 1), padding='SAME')
    #summary_layer(net, 'outlier_rej')

    # Final tanh activation
    net = tf.nn.tanh(net)
    return net
