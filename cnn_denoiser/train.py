import tensorflow as tf
import numpy as np
import scipy
import cv2
import input_data
import os
import glob
import importlib.util
from model_definitions import autoencoder_model as model

# Flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run', 'continue',
                            "Which operation to run. [continue|restart]")
tf.app.flags.DEFINE_string('num_iter', 100,
                            "How many iterations to run.")
tf.app.flags.DEFINE_string('model_name', 'tutorial_cnn',
                            "The name of the model in the model_definitions module")

# Paths
model_save_path = 'cnn_denoiser/trained_models/deblurring_model'
dataset_path = 'data/40nice'
logs_directory = './tensorboard_logs/'

# Parameters
image_width = 270
image_height = 90
batch_size = 30

# Hyperparameters
# alpha = 0.0001
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-1
N_steps_before_decay = 20
decay_rate = 0.9
alpha = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           N_steps_before_decay, decay_rate, staircase=True)

tf.summary.scalar('learning_rate', alpha)

# Load the model
model_file = os.path.dirname(os.path.abspath(__file__)) + "/model_definitions/networks/" + FLAGS.model_name + ".py"
spec = importlib.util.spec_from_file_location("model_definitions", model_file)
autoencoder_network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autoencoder_network)

network = model.initialise(image_width, image_height, autoencoder_network.autoencoder, batch_size, alpha, global_step)

# Load data
image_data = input_data.load_images(dataset_path, image_width,image_height)
batch_per_ep = len(image_data.imgs) // batch_size
if (len(image_data.imgs) == 0):
    e = 'No images were loaded - likely cause is wrong path: ' + dataset_path
    raise Exception(e)

# Logging
saver = tf.train.Saver()
files = glob.glob('tensorboard_logs/*')
for f in files:
    os.remove(f)
writer = tf.summary.FileWriter(logs_directory, graph=tf.get_default_graph())

# Train on training data, every epoch evaluate with same evaluation data
def train_model(sess, num_iter):
    output = open("output.txt", "w")
    count = 0
    print('Training model...')
    for i in range(num_iter):
        summary = None
        for batch_n in range(batch_per_ep):
            input_, blurred = image_data.next_batch(batch_size)
            _, cost, summary = sess.run([network.train_op, network.cost, network.summary_op], feed_dict={network.original: input_, network.corrupted: blurred})

            epoch_cost = 'Epoch: {} - cost= {:.8f}'.format(i, cost)
            output.write(epoch_cost + '\n')
            print(epoch_cost)

        count += 1
        if count % 10 == 0:
            writer.add_summary(summary, int(count / 10))
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
