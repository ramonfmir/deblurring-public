import tensorflow as tf
import numpy as np
from .model import Model

def initialise(image_width, image_height, autoencoder, batch_size, lr, global_step, training):
    input_size = batch_size if training else 1

    # Blurred image, input to the network.
    corrupted = tf.placeholder(tf.float32, (input_size, image_height, image_width, 3), name='corrupted')
    corrupted_greyscale = tf.reduce_mean(corrupted, axis=3,keep_dims = True)
    tf.summary.image('corrupted_greyscale', corrupted_greyscale, max_outputs=1)

    # Create the Autoencoder network (on a GPU).
    with tf.device('/gpu:0'):
        deblurred = autoencoder(corrupted, corrupted_greyscale, training)

    # Original, unblurred image to the network.
    original = tf.placeholder(tf.float32, (input_size, image_height, image_width, 3), name='original')
    original_greyscale = tf.reduce_mean(original, axis=3, keep_dims = True)
    tf.summary.image('original_greyscale', original_greyscale, max_outputs=1)

    # Calculate the loss and optimize the network.
    cost = tf.reduce_mean(tf.square(deblurred - original_greyscale))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, global_step=global_step)

    # Initialize the network.
    init = tf.global_variables_initializer()

    # Scalar summaries.
    tf.summary.scalar("cost", cost)
    tf.summary.image("deblurred", deblurred, max_outputs=1)
    summary_op = tf.summary.merge_all()

    return Model(train_op, cost, original, corrupted, deblurred, summary_op, init)
