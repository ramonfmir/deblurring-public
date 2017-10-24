import tensorflow as tf

# Parameters
channels = 1

def create_convolutional_layer(input,
               num_input_channels,
               conv_filter_size,
               num_filters):

    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer

def autoencoder(inputs, batch_size):
    # encoder
    print(inputs.shape)

    filter_1 = tf.Variable(tf.truncated_normal([5, 5, channels, 32], stddev=0.05))
    net = tf.nn.conv2d(inputs, filter_1, strides=[1, 3, 3, 1], padding='SAME')
    net = tf.nn.relu(net)
    print(net.shape)

    filter_2 = tf.Variable(tf.truncated_normal([5, 5, 32, 16], stddev=0.05))
    net = tf.nn.conv2d(net, filter_2, strides=[1, 2, 2, 1], padding='SAME')
    net = tf.nn.relu(net)
    print(net.shape)

    filter_3 = tf.Variable(tf.truncated_normal([5, 5, 16, 8], stddev=0.05))
    net = tf.nn.conv2d(net, filter_3, strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(net)
    print(net.shape)

    # decoder
    filter_t1 = tf.Variable(tf.truncated_normal([5, 5, 16, 8], stddev=0.05))
    net = tf.nn.conv2d_transpose(net, filter_t1, output_shape=[batch_size, 15, 45, 16], strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.relu(net)
    print(net.shape)

    filter_t2 = tf.Variable(tf.truncated_normal([5, 5, 32, 16], stddev=0.05))
    net = tf.nn.conv2d_transpose(net, filter_t2, output_shape=[batch_size, 30, 90, 32], strides=[1, 2, 2, 1], padding='SAME')
    net = tf.nn.relu(net)
    print(net.shape)

    filter_t3 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.05))
    net = tf.nn.conv2d_transpose(net, filter_t3, output_shape=[batch_size, 90, 270, channels], strides=[1, 3, 3, 1], padding='SAME')
    net = tf.nn.relu(net)

    print(net.shape)
    return net
