import tensorflow as tf
import sys
import os

from cnn_denoiser.model_definitions import autoencoder_model as model
from cnn_denoiser.model_definitions.networks import moussaka as autoencoder_network


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import scipy.misc
import glob

from PIL import Image

# Paths
model_save_path = 'cnn_denoiser/trained_models/deblurring_model'

# Parameters
image_width = 270
image_height = 90
batch_size = 20

# Lil hack
reuse = False

def save_clean(corrupted_images_path, clean_imges_path):
    imgs = glob.glob(os.path.join(corrupted_images_path, '*g'))

    global_step = tf.Variable(0, trainable=False)
    init_op = tf.global_variables_initializer()

    # Evaluate model
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        network = model.initialise(image_width, image_height, autoencoder_network.autoencoder, 1, 0.001, global_step, training=False)

        saver = tf.train.Saver()
        saver.restore(sess, model_save_path)

        for f in imgs:
            img = cv2.imread(f)

            img = cv2.resize(img, (image_width, image_height), 0, 0, cv2.INTER_CUBIC)

            img = [np.asarray(np.multiply(img.astype(np.float32), 1.0 / 255.0))]

            clean = sess.run([network.deblurred], feed_dict={network.corrupted: img, network.original: img})[0]
            clean = clean[0, ..., 0]

            scipy.misc.imsave(clean_images_path + "/" + os.path.basename(f), clean)


if __name__ == "__main__":
    corrupted_images_path = sys.argv[1]
    clean_images_path = sys.argv[2]

    save_clean(corrupted_images_path, clean_images_path)
