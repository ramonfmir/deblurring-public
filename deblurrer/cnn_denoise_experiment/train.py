import tensorflow as tf
import numpy as np
import sys
# from tensorflow.examples.tutorials.mnist import input_data
import input_data as input_data #input_data
import conv_decov_model as model
import matplotlib.pyplot as plt


# Paths
model_save_path = './trained_models_emnist/autoencoder_model' # './trained_models/autoencoder_model'
mnist_data_path = './EMNIST_data/' # './MNIST_data/'

# Parameters
corruption_level = 0.2

# Hyperparameters
alpha = 0.01

# Load MNIST data
mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

saver = tf.train.Saver()

# Train on training data, every epoch evaluate with same evaluation data
def train_model(sess):
    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            # (128, 784)
            input_ = trX[start:end]
            noise_mask = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(model.train_op, feed_dict={model.original: input_, model.corrupted: noise_mask * input_})
            rec = sess.run(model.deblurred, feed_dict={model.original: input_, model.corrupted: noise_mask * input_})
            # print('Deblurred: ', rec[0])
            # plt.imshow(rec[0].reshape((28, 28)), cmap='gray')
            # plt.show()

        noise_mask = np.random.binomial(1, 1 - corruption_level, teX.shape)
        print('Cost: ', sess.run(model.cost, feed_dict={model.original: teX, model.corrupted: noise_mask * teX}))

        saver.save(sess, model_save_path)

# Run training / viewing
with tf.Session() as sess:
    if (sys.argv[1] == 'restart'):
        tf.initialize_all_variables().run()
    else:
        saver.restore(sess, model_save_path)
    train_model(sess)
