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

from PIL import Image

# Paths
model_save_path = 'cnn_denoiser/trained_models/deblurring_model'

# Parameters
image_width = 270
image_height = 90
batch_size = 20

global_step = tf.Variable(0, trainable=False)

init_op = tf.global_variables_initializer()


if __name__ == "__main__":
    file_name = sys.argv[1]
    img = cv2.imread(file_name)
    img = cv2.resize(img, (image_width, image_height), 0, 0, cv2.INTER_CUBIC)

    # Evaluate model
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        network = model.initialise(image_width, image_height, autoencoder_network.autoencoder, 1, 0.001, global_step, training=False)

        saver = tf.train.Saver()
        saver.restore(sess, model_save_path)

        img = [np.asarray(np.multiply(img.astype(np.float32), 1.0 / 255.0))]
        
        clean = sess.run([network.deblurred], feed_dict={network.corrupted: img, network.original: img})[0]
        #print(clean)
        #clean = 
        #result = clean.reshape((1, 90, 270, 1))

    print(clean.shape)
    clean = clean[0, ..., 0]
    print(clean.shape)

    #cv2.imshow('hey', clean)
    scipy.misc.imsave('result.jpg', clean)
