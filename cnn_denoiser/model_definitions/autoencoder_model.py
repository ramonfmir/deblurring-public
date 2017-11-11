import tensorflow as tf
import numpy as np
from .model import Model

def initialise(image_width, image_height, autoencoder, batch_size, lr, global_step):
    # original, unblurred image to the network
    original = tf.placeholder(tf.float32, (batch_size, image_height, image_width, 3))
    original_greyscale = tf.reduce_mean(original, axis=3, keep_dims = True)
    tf.summary.image('original_greyscale', original_greyscale, max_outputs=1)

    # blurred image, input to the network
    corrupted = tf.placeholder(tf.float32, (batch_size, image_height, image_width, 3))
    corrupted_greyscale = tf.reduce_mean(corrupted, axis=3,keep_dims = True)
    tf.summary.image('corrupted_greyscale', corrupted_greyscale, max_outputs=1)

    deblurred = autoencoder(corrupted_greyscale, batch_size)  # create the Autoencoder network

    # calculate the loss and optimize the network
    cost = tf.reduce_mean(tf.square(deblurred - original_greyscale))  # claculate the mean square error loss
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, global_step=global_step)
    # initialize the network
    init = tf.global_variables_initializer()

    # Scalar summaries
    tf.summary.scalar("cost", cost)
    tf.summary.image("deblurred", deblurred, max_outputs=1)
    summary_op = tf.summary.merge_all()

    return Model(train_op, cost, original, corrupted, deblurred, summary_op, init)
