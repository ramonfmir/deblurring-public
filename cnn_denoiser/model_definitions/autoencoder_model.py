import tensorflow as tf
import numpy as np
from .model import Model

gpus = 2

def combine_gradients(gradients):
    average_grads = []
    for grad_and_vars in zip(*gradients):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            print(g)
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

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
    all_costs = []
    reuse_vars = False
    for gpu in range(gpus):
        with tf.device('/gpu:%d' % gpu):
            corrupted_greyscale_minibatch = corrupted_greyscale[gpu * minibatch_size : (gpu + 1) * minibatch_size]
            original_greyscale_minibatch = original_greyscale[gpu * minibatch_size : (gpu + 1) * minibatch_size]

            # Create the autoencoder network.
            deblurred = autoencoder(corrupted_greyscale_minibatch, reuse_vars)

            # Cost is squared error.
            loss_op = tf.reduce_mean(tf.square(deblurred - original_greyscale_minibatch))
            all_costs.append(loss_op)

            optimiser = tf.train.AdamOptimizer(learning_rate=lr)
            grads = optimiser.compute_gradients(loss_op)

            reuse_vars = True
            all_grads.append(grads)
    
    all_grads = combine_gradients(all_grads)
    train_op = optimiser.apply_gradients(all_grads)

    cost = 0

    # initialize the network
    init = tf.global_variables_initializer()

    # Scalar summaries
    tf.summary.scalar("cost", cost)
    tf.summary.image("deblurred", deblurred, max_outputs=1)
    summary_op = tf.summary.merge_all()

    return Model(train_op, cost, original, corrupted, deblurred, summary_op, init)

