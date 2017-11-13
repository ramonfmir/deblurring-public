import tensorflow as tf
import sys

from model_definitions import autoencoder_model as model
from model_definitions.networks import conv_deconv as autoencoder_network

import input_data
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import scipy.misc
import glob

# Paths
model_save_path = 'cnn_denoiser/trained_models/deblurring_model'
# dataset_path = 'data/4000unlabeledLP_same_dims_scaled'
dataset_path = 'data/100labeledLPforvalidation'
logs_directory = './evaluate_logs/'

# Parameters
image_width = 270
image_height = 90
batch_size = 30
num_test = int(100 / batch_size)

global_step = tf.Variable(0, trainable=False)

# Method to show results visually
def show_encoding(sess, writer, original, network):
    summary_orig = sess.run(network.summary_op, feed_dict={network.corrupted: original, network.original: original})
    writer.add_summary(summary_orig, 0)

# Evaluate model
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # Load graph
    network = model.initialise(image_width, image_height, autoencoder_network.autoencoder, batch_size, 0.001, global_step)

    saver = tf.train.Saver()
    saver.restore(sess, model_save_path)

    # Load MNIST data
    image_data = input_data.load_images(dataset_path, image_width,image_height)

    # Test images used for examples in README
    writer = tf.summary.FileWriter(logs_directory, graph=tf.get_default_graph())

    for img in range(num_test):
        test_ori_images, _ = image_data.next_batch(batch_size)
        show_encoding(sess, writer, test_ori_images, network)
