from deblurrer.cnn_autoencoder.model_definitions import autoencoder_model as model
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.framework import ops

# Flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run', 'continue',
                            "Which operation to run. [continue|restart]")
tf.app.flags.DEFINE_string('num_iter', 100,
                            "How many iterations to run.")
tf.app.flags.DEFINE_string('model_name', 'tutorial_cnn',
                            "The name of the model in the model_definitions module")

# Paths
model_save_path = './trained_models/deblurring_model'
dataset_path = 'data/4000unlabeledLP_same_dims_scaled'
logs_directory = './logs/'

# Parameters
image_width = 270
image_height = 90
batch_size = 40

# Hyperparameters
alpha = 0.005

# Load the model
model_file = os.path.dirname(os.path.abspath(__file__)) + "/model_definitions/" + FLAGS.model_name + ".py"
spec = importlib.util.spec_from_file_location("model_definitions", model_file)
autoencoder_network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autoencoder_network)

# Hyperparameters
alpha = 0.005

class test_network_cost(tf.test.TestCase):
    def test_network_cost(self):
        with self.test_session():
            original = tf.placeholder(tf.float32, (batch_size, image_height, image_width, 3))
            original_greyscale = tf.reduce_mean(original, axis=3, keep_dims = True)
            tf.summary.image('original_greyscale', original_greyscale, max_outputs=2)

            network = model.initialise(image_width, image_height, autoencoder_network.autoencoder, batch_size, alpha)

            # calculate the loss and optimize the network
            self.assertEqual(network.original.shape, original.shape)

if __name__ == '__main__':
    tf.test.main()
