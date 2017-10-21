import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy
import input_data
import blurrer
import os
import glob

sys.path.insert(0, './model_definitions/')
import cnn_trial_model as model

# Flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run', 'restart',
                            "Which operation to run. [continue|restart]")

# Paths
model_save_path = './trained_models/deblurring_model'
dataset_path = '../../data/4000unlabeledLP_same_dims_scaled'
logs_directory = './logs/'

# Parameters
image_width = 270
image_height = 90
batch_size = 100
num_iter = 10

# Hyperparameters
alpha = 0.01

# Load the model
network = model.initialise(image_width, image_height, alpha)

# Load data
image_data, _ = input_data.load_images(dataset_path, image_width,image_height)
num_train = int(0.996 * len(image_data))
trX, teX = image_data[:num_train], image_data[num_train:]

# Logging
saver = tf.train.Saver()
files = glob.glob('logs/*')
for f in files:
    os.remove(f)
writer = tf.summary.FileWriter(logs_directory, graph=tf.get_default_graph())

# Train on training data, every epoch evaluate with same evaluation data
def train_model(sess):
    count = 0
    for i in range(num_iter):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            input_ = trX[start:end]
            blurred = blurrer.blur_all(input_) # (100, img)
            _, cost, summary = sess.run([network.train_op, network.cost, network.summary_op], feed_dict={network.original: input_, network.corrupted: blurred})
            count += 1
            writer.add_summary(summary, count)
            print('Epoch: {} - cost= {:.8f}'.format(i, cost))

        saver.save(sess, model_save_path)


# Run continue training / restart training
def main(argv=None):
    with tf.Session() as sess:
        if FLAGS.run == 'continue':
            saver.restore(sess, model_save_path)
        elif FLAGS.run == 'restart':
            network.init.run()
        else:
            saver.restore(sess, model_save_path)
        train_model(sess)

if __name__ == "__main__":
    tf.app.run()
