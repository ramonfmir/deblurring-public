import tensorflow as tf
import cnn_trial_model as model
# from tensorflow.examples.tutorials.mnist import input_data
import input_data
import numpy as np
import matplotlib.pyplot as plt

# Paths
model_save_path = './trained_models/deblurring_model' # './trained_models/autoencoder_model'
dataset_path = '../../data/4000unlabeledLP_same_dims_scaled'

# Parameters
corruption_level = 0.3
image_width = 270
image_height = 90

# Method to show results visually
def show_encoding(sess, imgs, network):
    blurred = model.add_noise(imgs)
    recon_img = sess.run([network.deblurred], feed_dict={network.corrupted: blurred})[0]

    for i in range(len(imgs)):
        plt.figure(1)
        plt.title('Reconstructed Images')
        plt.imshow(recon_img[i, ..., 0], cmap='gray')
        plt.figure(2)
        plt.title('Input Images')
        gray_scale = tf.reduce_mean(blurred, axis=3, keep_dims=True)
        plt.imshow(sess.run(gray_scale)[i, ..., 0],cmap = 'gray')
        plt.show()

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
