import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
import input_data
import os
import glob

from model_definitions import autoencoder_model as model
from model_definitions import tutorial_cnn as autoencoder_network

# Flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run', 'continue',
                            "Which operation to run. [continue|restart]")
tf.app.flags.DEFINE_string('num_iter', 100,
                            "How many iterations to run.")

# Paths
model_save_path = './trained_models/deblurring_model'
dataset_path = '../../data/4000unlabeledLP_same_dims_scaled'
logs_directory = './logs/'

# Parameters
image_width = 270
image_height = 90
batch_size = 25

# Hyperparameters
alpha = 0.01

# Load the model
network = model.initialise(image_width, image_height, autoencoder_network.autoencoder, batch_size, alpha)

# Load data
image_data = input_data.load_images(dataset_path, image_width,image_height)
batch_per_ep = len(image_data.imgs) // batch_size
if (len(image_data.imgs) == 0):
    e = 'No images were loaded - likely cause is wrong path: ' + dataset_path
    raise Exception(e)

# Logging
saver = tf.train.Saver()
files = glob.glob('logs/*')
for f in files:
    os.remove(f)
writer = tf.summary.FileWriter(logs_directory, graph=tf.get_default_graph())

# Train on training data, every epoch evaluate with same evaluation data
def train_model(sess, num_iter):
    output = open("output.txt", "w")
    count = 0
    for i in range(num_iter):
        for batch_n in range(batch_per_ep):
            input_,blurred = image_data.next_batch(batch_size)
            _, cost, summary = sess.run([network.train_op, network.cost, network.summary_op], feed_dict={network.original: input_, network.corrupted: blurred})
            count += 1
            writer.add_summary(summary, count)

            epoch_cost = 'Epoch: {} - cost= {:.8f}'.format(i, cost)
            output.write(epoch_cost + '\n')
            print(epoch_cost)

        saver.save(sess, model_save_path)

    output.close()

# Run continue training / restart training
def main(argv=None):
    with tf.Session() as sess:
        if FLAGS.run == 'continue':
            saver.restore(sess, model_save_path)
        elif FLAGS.run == 'restart':
            network.init.run()
        else:
            saver.restore(sess, model_save_path)
        train_model(sess, int(FLAGS.num_iter))

if __name__ == "__main__":
    tf.app.run()
