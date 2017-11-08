import tensorflow as tf
import numpy as np
from .model import Model

gpus = 4

def initialise(image_width, image_height, autoencoder, batch_size, lr=0.01):
    # original, unblurred image to the network
    original = tf.placeholder(tf.float32, (batch_size, image_height, image_width, 3))
    original_greyscale = tf.reduce_mean(original, axis=3, keep_dims = True)
    tf.summary.image('original_greyscale', original_greyscale, max_outputs=1)

    # blurred image, input to the network
    corrupted = tf.placeholder(tf.float32, (batch_size, image_height, image_width, 3))
    corrupted_greyscale = tf.reduce_mean(corrupted, axis=3,keep_dims = True)
    tf.summary.image('corrupted_greyscale', corrupted_greyscale, max_outputs=1)

    minibatch_size = batch_size // gpus
    all_grads = []
    all_costs = 0.0
    reuse_vars = False
    for gpu in range(gpus):
        with tf.device('/gpu:%d' % gpu):
            corrupted_greyscale_minibatch = corrupted_greyscale[gpu * minibatch_size : (gpu + 1) * minibatch_size]
            original_greyscale_minibatch = original_greyscale[gpu * minibatch_size : (gpu + 1) * minibatch_size]

            # Create the autoencoder network.
            deblurred = autoencoder(corrupted_greyscale_minibatch, reuse_vars)

            # Cost is squared error.
            loss_op = tf.reduce_mean(tf.square(deblurred - original_greyscale_minibatch))
            all_costs += loss_op

            #optimiser = tf.train.AdamOptimizer(learning_rate=lr)
            #grads = optimiser.compute_gradients(loss_op)

            reuse_vars = True
            #all_grads.append(grads)
    
    #all_grads = combine_gradients(all_grads)
    #train_op = optimiser.apply_gradients(all_grads)
    #print(all_costs)

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(all_costs)

    # initialize the network
    init = tf.global_variables_initializer()

    # Scalar summaries
    tf.summary.scalar("cost", loss_op)
    tf.summary.image("deblurred", deblurred, max_outputs=1)
    summary_op = tf.summary.merge_all()

    return Model(train_op, all_costs, original, corrupted, deblurred, summary_op, init)

