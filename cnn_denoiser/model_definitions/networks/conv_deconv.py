import tensorflow as tf

# Parameters
channels = 1

def summary_layer(net, name):
    tf.summary.image(name, tf.expand_dims(tf.transpose(net, [3, 0, 1, 2])[0], 3), max_outputs=1)
    print(net.shape)

def conv_layer_dropout(net, layer, out_channels, filter_dims, strides, padding, name, dropout, act_f = tf.nn.relu):
    net = tf.layers.dropout(net, dropout)
    return conv_layer(net, layer, out_channels, filter_dims, strides, padding, name, act_f)

def conv_layer(net, layer, out_channels, filter_dims, strides, padding, name, act_f = tf.nn.relu):
    net = layer(net, out_channels, filter_dims, strides=strides, padding=padding)
    net = act_f(net)
    summary_layer(net, name)
    return net

def autoencoder(inputs, batch_size, dropout=0.65):
    # Encoder
    net =      conv_layer(inputs, tf.layers.conv2d, 256, [5, 5], (3, 3), 'SAME', 'conv1')
    net = conv_layer_dropout(net, tf.layers.conv2d, 128, [5, 5], (2, 2), 'SAME', 'conv2', dropout)
    net = conv_layer_dropout(net, tf.layers.conv2d, 64 , [5, 5], (1, 1), 'SAME', 'conv3', dropout)

    # Decoder
    net = conv_layer_dropout(net, tf.layers.conv2d_transpose, 64 , [5, 5], (1, 1), 'SAME', 'deconv1', dropout)
    net = conv_layer_dropout(net, tf.layers.conv2d_transpose, 128, [5, 5], (2, 2), 'SAME', 'deconv2', dropout)
    net = conv_layer_dropout(net, tf.layers.conv2d_transpose, channels, [5, 5], (3, 3), 'SAME', 'deconv3', dropout)

    # Final tanh activation
    net = tf.nn.tanh(net)
    return net
