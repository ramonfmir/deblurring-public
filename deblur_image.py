import tensorflow as tf
import sys

from cnn_denoiser.model_definitions import autoencoder_model as model
from cnn_denoiser.model_definitions.networks import moussaka as autoencoder_network


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import scipy.misc
import glob

# Paths
model_save_path = 'cnn_denoiser/trained_models/deblurring_model'

# Parameters
image_width = 270
image_height = 90
batch_size = 20

global_step = tf.Variable(0, trainable=False)

if __name__ == "__main__":
    file_name = sys.argv[1]
    img = cv2.imread(file_name)
    img = cv2.resize(img, (image_width, image_height), 0, 0, cv2.INTER_CUBIC)

    # Evaluate model
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        network = model.initialise(image_width, image_height, autoencoder_network.autoencoder, 1, 0.001, global_step, training=False)

        saver = tf.train.Saver()
        saver.restore(sess, model_save_path)

        original = [np.asarray(np.multiply(img.astype(np.float32), 1.0 / 255.0))]
        sess.run(network.summary_op, feed_dict={network.corrupted: original,
                                                network.original: original})

        result = np.array(network.corrupted[0])

    scipy.misc.imsave('result.jpg', result.astype(np.uint8))
