import tensorflow as tf
import sys

from model_definitions import autoencoder_model as model
from model_definitions.networks import moussaka as autoencoder_network

import input_data
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import scipy.misc
import glob

# Paths
model_save_path = 'cnn_denoiser/trained_models/deblurring_model'
dataset_path = 'data/100labeledLPforvalidation'
logs_directory = './evaluate_logs/'

# Parameters
image_width = 270
image_height = 90
batch_size = 20

global_step = tf.Variable(0, trainable=False)

# Method to show results visually
def show_encoding(sess, writer, original, network):
    summary_orig = sess.run(network.summary_op, feed_dict={network.corrupted: original, network.original: original})
    writer.add_summary(summary_orig, 0)

# Evaluate model
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # Load graph
    network = model.initialise(image_width, image_height, autoencoder_network.autoencoder, batch_size, 0.001, global_step, training=False)

    saver = tf.train.Saver()
    saver.restore(sess, model_save_path)

    dataset = input_data.load_images(dataset_path, image_width,image_height)

    writer = tf.summary.FileWriter(logs_directory, graph=tf.get_default_graph())

    for img in dataset.imgs:
        show_encoding(sess, writer, [dataset.normalise_image(img)], network)
