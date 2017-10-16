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
def show_encoding(sess, imgs, model):
    noise_mask = np.random.normal(0, 1 - corruption_level, imgs.shape)
    recon_img = sess.run([model.deblurred], feed_dict={model.corrupted: noise_mask * imgs})[0]
    print("After weed sess")
    plt.imshow(recon_img[0], cmap='gray')
    plt.show()

# Evaluate model
with tf.Session() as sess:
    saver = tf.train.Saver()
    network = model.initialise(image_width, image_height, 0.1)
    # tf.initialize_all_variables().run()
    saver.restore(sess, model_save_path)

    # Accessing the default graph which we have restored
    #graph = tf.get_default_graph()

    # Load MNIST data
    test_images, _ = input_data.load_images(dataset_path, image_width,image_height)
    test_images = test_images[:10]

    # Test images used for examples in README
    show_encoding(sess, test_images, network)
