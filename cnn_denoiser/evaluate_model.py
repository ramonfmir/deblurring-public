import tensorflow as tf
import sys
sys.path.insert(0, './model_definitions/')

from model_definitions import autoencoder_model as model
from model_definitions import tutorial_cnn as autoencoder_network

import input_data
import numpy as np
import blurrer
import matplotlib.pyplot as plt
from skimage import color
import scipy.misc
import glob
sys.path.insert(0, "../..")
import blurrer.blurring_experiment as blurrer

# Paths
<<<<<<< HEAD:cnn_autoencoder/evaluate_model.py
model_save_path = './trained_models/deblurring_model' # './trained_models/autoencoder_model'
dataset_path = 'data/4000unlabeledLP_same_dims_scaled'
images_path = './results/'
=======
model_save_path = 'deblurrer/cnn_autoencoder/trained_models/deblurring_model'
dataset_path = 'data/100labeledLPforvalidation_same_dims_scaled'
images_path = 'deblurrer/cnn_autoencoder/results/'
>>>>>>> 47fdb036f8f3ab1b27d87eda526495861cff4d4b:deblurrer/cnn_autoencoder/evaluate_model.py

# Parameters
image_width = 270
image_height = 90
batch_size = 10

# Method to show results visually
def show_encoding(sess, original, blurred, network):
    recon_img = sess.run([network.deblurred], feed_dict={network.corrupted: blurred})[0]

    for i in range(len(original)):
        scipy.misc.imsave(images_path + str(i) + '/deblurred.jpg', recon_img[i, ..., 0])
        scipy.misc.imsave(images_path + str(i) + '/blurred.jpg', blurred[i])
        scipy.misc.imsave(images_path + str(i) + '/original.jpg', original[i])

# Evaluate model
with tf.Session() as sess:
    # Load graph
    network = model.initialise(image_width, image_height, autoencoder_network.autoencoder, batch_size)

    saver = tf.train.Saver()
    saver.restore(sess, model_save_path)

    # Load MNIST data
    image_data = input_data.load_images(dataset_path, image_width,image_height)
    test_ori_images, test_corr_images = image_data.next_batch(batch_size)

    # Test images used for examples in README
    show_encoding(sess, test_ori_images, test_ori_images, network)