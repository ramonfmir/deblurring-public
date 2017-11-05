import tensorflow as tf

# Parameters
channels = 1

def create_filters(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(dim):
    #TODO
    return tf.Variable(tf.constant(0.05, shape=[dim]))

def create_convolutional_layer(input_, num_input_channels, kernel_x, kernel_y,
               num_output_channels, strides):

    filter_ = create_filters(shape=[kernel_x, kernel_y, num_input_channels, num_output_channels])
    biases = create_biases(num_output_channels)
    net = tf.nn.conv2d(input=input_, filter=filter_, strides=[1, 1, 1, 1], padding='SAME')

    #layer += biases
    net = tf.nn.relu(net)

    return net

def create_convolutional_transposed_layer(input_, num_input_channels, kernel_x, kernel_y,
               num_output_channels, strides):

    filter_ = create_filters(shape=[kernel_x, kernel_y, num_input_channels, num_output_channels])
    biases = create_biases(num_output_channels)
    net = tf.nn.conv2d(input=input_, filter=filter_, strides=[1, 1, 1, 1], padding='SAME')

    #layer += biases
    net = tf.nn.relu(net)

    return net

def autoencoder(inputs, batch_size):
    # encoder
    filter_1 = tf.Variable(tf.truncated_normal([5, 5, channels, 32], stddev=0.05))
    net = tf.nn.conv2d(inputs, filter_1, strides=[1, 3, 3, 1], padding='SAME')
    net = tf.nn.relu(net)

    filter_2 = tf.Variable(tf.truncated_normal([5, 5, 32, 16], stddev=0.05))
    net = tf.nn.conv2d(net, filter_2, strides=[1, 2, 2, 1], padding='SAME')
    net = tf.nn.relu(net)

    filter_3 = tf.Variable(tf.truncated_normal([5, 5, 16, 8], stddev=0.05))
    net = tf.nn.conv2d(net, filter_3, strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(net)

    # decoder
    filter_t1 = tf.Variable(tf.truncated_normal([5, 5, 16, 8], stddev=0.05))
    net = tf.nn.conv2d_transpose(net, filter_t1, output_shape=[batch_size, 15, 45, 16], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(net)

    filter_t2 = tf.Variable(tf.truncated_normal([5, 5, 32, 16], stddev=0.05))
    net = tf.nn.conv2d_transpose(net, filter_t2, output_shape=[batch_size, 30, 90, 32], strides=[1, 2, 2, 1], padding='SAME')
    net = tf.nn.relu(net)

    filter_t3 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.05))
    net = tf.nn.conv2d_transpose(net, filter_t3, output_shape=[batch_size, 90, 270, channels], strides=[1, 3, 3, 1], padding='SAME')
    net = tf.nn.relu(net)

    return net
