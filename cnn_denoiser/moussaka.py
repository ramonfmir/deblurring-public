import tensorflow as tf
import os
import sys
import input_data
import glob


""" Hyperparameters """
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
N_steps_before_decay = 10
decay_rate = 0.8
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           N_steps_before_decay, decay_rate, staircase=True)

tf.summary.scalar('learning_rate', learning_rate)

l1_regularization_strength = 0.01
l2_regularization_strength = 0.01

num_iter = 500


""" Get all images """
dataset_path = "../data/40nice/"
image_width = 270
image_height = 90
batch_size = 20

image_data = input_data.load_images(dataset_path, image_width,image_height)
batch_per_ep = len(image_data.imgs) // batch_size
if (len(image_data.imgs) == 0):
    e = 'No images were loaded - likely cause is wrong path: ' + dataset_path
    raise Exception(e)


""" Set up tensorboard_logs """
logs_directory = "logs"
# saver = tf.train.Saver()
files = glob.glob('%s/*' % logs_directory)
for f in files:
    os.remove(f)


""" Utils """
def saveImage(name, image):
    tf.summary.image(name, image, max_outputs=1) # max_outputs?

def saveLayerImage(name, layer):
    tf.summary.image(name, tf.expand_dims(tf.transpose(layer, [3, 0, 1, 2])[0], 3), max_outputs=1)


""" Input placeholders for original and blurred image """
original = tf.placeholder(tf.float32, (batch_size, image_height, image_width, 3))
original_greyscale = tf.reduce_mean(original, axis=3, keep_dims = True)
saveImage('original_greyscale', original_greyscale)

# blurred image, input to the network
corrupted = tf.placeholder(tf.float32, (batch_size, image_height, image_width, 3))
corrupted_greyscale = tf.reduce_mean(corrupted, axis=3, keep_dims = True)
saveImage('corrupted_greyscale', corrupted_greyscale)


""" Build the network """
# Layer 1
net = tf.contrib.layers.conv2d(corrupted_greyscale, 32, [5, 5], stride=3, padding='SAME')
saveLayerImage('conv1', net)

net = tf.contrib.layers.conv2d(net, 32, [5, 5], stride=1, padding='SAME')
saveLayerImage('code_layer', net)

net = tf.contrib.layers.conv2d_transpose(net, 32, [5, 5], stride=1, padding='SAME')
saveLayerImage('deconv1', net)

net = tf.contrib.layers.conv2d_transpose(net, 32, [6, 6], stride=3, padding='SAME')
saveLayerImage('deconv2', net)

deblurred = tf.contrib.layers.conv2d(net, 1, [5, 5], stride=1, padding='SAME', activation_fn=tf.nn.tanh)
tf.summary.image("deblurred", deblurred, max_outputs=1)

cost = tf.reduce_mean(tf.square(deblurred - original_greyscale))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
# train_op = tf.train.ProximalGradientDescentOptimizer(learning_rate, l1_regularization_strength, l2_regularization_strength).minimize(cost, global_step=global_step)
# initialize the network
init = tf.global_variables_initializer()

# Scalar summaries
tf.summary.scalar("cost", cost)
summary_op = tf.summary.merge_all()

writer = tf.summary.FileWriter(logs_directory, graph=tf.get_default_graph())

# Train on training data, every epoch evaluate with same evaluation data
count = 0.0
with tf.Session() as sess:
    print('Training model...')
    sess.run(init)
    for i in range(num_iter):
        for batch_n in range(batch_per_ep):
            input_, blurred = image_data.next_batch(batch_size)
            step, _, cost_, summary = sess.run([global_step, train_op, cost, summary_op], feed_dict={corrupted: blurred, original: input_})
            writer.add_summary(summary, count)
            count += 1
            print('Epoch: {} - cost= {:.8f}'.format(i, cost_*100))

#         summary = session.run(merged,
#                               options=run_options,
#                               run_metadata=run_metadata)
#         writer.add_run_metadata(run_metadata, 'step%d' % i)
#         writer.add_summary(summary, global_step=i)
