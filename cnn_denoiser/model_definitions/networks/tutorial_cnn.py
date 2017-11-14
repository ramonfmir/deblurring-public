import tensorflow as tf

# Parameters
channels = 1

def summary_layer(net, name):
    tf.summary.image(name, tf.expand_dims(tf.transpose(net, [3, 0, 1, 2])[0], 3), max_outputs=1)
    print(net.shape)


def autoencoder(inputs, batch_size, dropout=0.65):
    inits = tf.constant(0.5)
    # Encoder
    pre_lay1 = tf.layers.conv2d(inputs, 32, [3, 3], strides=(1, 1), padding='SAME')

    lay1 = tf.layers.conv2d(pre_lay1, 32, [3, 3], strides=(1, 1), padding='SAME')
    lay1 = tf.nn.lrn(lay1)
    k1 = tf.Variable(inits)
    tf.summary.scalar('k1', k1)
    lay1_ = k1 * lay1 + (1-k1) * pre_lay1
    summary_layer(lay1, 'conv1')
    summary_layer(lay1_, 'conv1_')

    # net = tf.layers.dropout(net, dropout)
    lay2 = tf.layers.conv2d(lay1_, 32, [5, 5], strides=(1, 1), padding='SAME')
    lay2 = tf.nn.lrn(lay2)
    k2 = tf.Variable(inits)
    tf.summary.scalar('k2', k2)
    lay2_ = k2 * lay2 + (1-k2)*lay1_
    summary_layer(lay2, 'conv2')
    summary_layer(lay2_, 'conv2_')

    # net = tf.layers.dropout(net, dropout)
    lay3 = tf.layers.conv2d(lay2_, 32, [3, 3], strides=(1, 1), padding='SAME')
    lay3 = tf.nn.lrn(lay3)
    k3 = tf.Variable(inits)
    tf.summary.scalar('k3', k3)
    lay3_ = k3 * lay3 + (1-k3)*lay2_
    summary_layer(lay3, 'conv3')
    summary_layer(lay3_, 'conv3_')

    # net = tf.layers.dropout(net, dropout)
    lay4 = tf.layers.conv2d(lay3_, 32, [3, 3], strides=(1, 1), padding='SAME')
    lay4 = tf.nn.lrn(lay4)
    k4 = tf.Variable(inits)
    tf.summary.scalar('k4', k4)
    lay4_ = k4 * lay4 + (1-k4)*lay3_
    summary_layer(lay3, 'conv4')
    summary_layer(lay3_, 'conv4_')

    # net = tf.layers.dropout(net, dropout)
    net = tf.layers.conv2d(lay4_, 1, [3, 3], strides=(1, 1), padding='SAME')
    net = tf.nn.lrn(net)
    summary_layer(net, 'conv5')

    return net
