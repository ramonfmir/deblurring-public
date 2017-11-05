import tensorflow as tf
import model

# Initial weights need to be built and saved so that their transpose may be
# used for the reflected part of the network
def build_weights(layers):
    weights = []
    for i in range(len(layers) - 1):
        weights_shape = [layers[i], layers[i+1]]
        W_init = tf.random_uniform(shape=weights_shape,
                                   minval=-0.5,
                                   maxval=0.5)
        weights += [W_init]
    return weights

def build_layers(weights, Y, name):
    for i in range(len(weights)):
        W_init = weights[i]

        W = tf.Variable(W_init)
        b = tf.Variable(tf.zeros([weights[i].shape[1].value]))
        Y = tf.nn.sigmoid(tf.matmul(Y, W) + b) # CHANGE TO RELU
    return Y

def initialise(image_width, image_height, learning_rate):
    # Parameters
    n_visible = image_width * image_height * 3
    code_layer_size = 20
    layers = [n_visible, 500, code_layer_size]

    # Input placeholders for original image and its corrupted version
    original = tf.placeholder("float", [None, image_width,image_height,3])
    corrupted = tf.placeholder("float", [None, image_width,image_height,3])
    original_flat = tf.reshape(original, [ 128,n_visible])
    corrupted_flat = tf.reshape(corrupted, [ 128,n_visible])

    # Build the encoder
    weights = build_weights(layers)
    encoding = build_layers(weights, corrupted_flat, 'Encoding Layer')

    # Used the transpose weights to initialise decoder
    decoding_weights = [tf.transpose(x) for x in weights[::-1]]
    decoding = build_layers(decoding_weights, encoding, 'Decoding Layer')

    # Cost is defined as error between original and reproduced
    cost = tf.reduce_sum(tf.pow(decoding - original_flat, 2))  # minimize squared error
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # construct an optimizer

    init = tf.global_variables_initializer()
    return model.Model(train_op, cost, original_flat, corrupted_flat, init)
