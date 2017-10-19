import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy
import model
import input_data
import blurrer

# ---------- Here choose your model ----------
# import conv_decov_model as model
import cnn_trial_model as model

# Paths
model_save_path = './trained_models/deblurring_model'
dataset_path = '../../data/4000unlabeledLP_same_dims_scaled'

# Parameters
image_width = 270
image_height = 90
batch_size = 100
num_iter = 5

# Hyperparameters
alpha = 0.01

# Load model
network = model.initialise(image_width, image_height, alpha)

# Load data
image_data, _ = input_data.load_images(dataset_path, image_width,image_height)
num_train = int(0.996 * len(image_data))
trX, teX = image_data[:num_train], image_data[num_train:]

saver = tf.train.Saver()

# Train on training data, every epoch evaluate with same evaluation data
def train_model(sess):
    for i in range(num_iter):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            input_ = trX[start:end]
            blurred = blurrer.blur_all(input_) # (100, img)
            _,cost = sess.run([network.train_op, network.cost], feed_dict={network.original: input_, network.corrupted: blurred})
            print('Epoch: {} - cost= {:.8f}'.format(i, cost))

        saver.save(sess, model_save_path)


# Run training / viewing
with tf.Session() as sess:
    if (len(sys.argv) != 0 and sys.argv[1] == 'restart'):
        network.init.run()
    else:
        saver.restore(sess, model_save_path)
    train_model(sess)
