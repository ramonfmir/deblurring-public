import tensorflow as tf
import sys
sys.path.insert(0, './model_definitions/')
import cnn_trial_model as model
import input_data
import numpy as np
import blurrer
import matplotlib.pyplot as plt
from skimage import color
import scipy.misc
import glob

# Paths
model_save_path = './trained_models/deblurring_model' # './trained_models/autoencoder_model'
dataset_path = '../../data/4000unlabeledLP_same_dims_scaled'
images_path = './results/'

# Parameters
corruption_level = 0.3
image_width = 270
image_height = 90

# Method to show results visually
def show_encoding(sess, imgs, network):
    blurred = blurrer.blur_all(imgs)
    print(np.shape(blurred))
    recon_img = sess.run([network.deblurred], feed_dict={network.corrupted: blurred})[0]

    for i in range(len(imgs)):
        scipy.misc.imsave(images_path + 'deblurred.jpg', recon_img[i, ..., 0])
        scipy.misc.imsave(images_path + 'blurred.jpg', blurred[i])
        scipy.misc.imsave(images_path + 'original.jpg', imgs[i])

# Evaluate model
with tf.Session() as sess:
    # Load graph
    network = model.initialise(image_width, image_height)

    saver = tf.train.Saver()
    saver.restore(sess, model_save_path)

    # Load MNIST data
    test_images, _ = input_data.load_images(dataset_path, image_width,image_height)
    test_images = test_images[:10]

    # Test images used for examples in README
    show_encoding(sess, test_images, network)