import tensorflow.contrib.layers as lays
import tensorflow as tf

# Parameters
channels = 1

def autoencoder(inputs, gpu):
	with tf.variable_scope('VarScope_%d' % gpu):
	    # Encoder (convolutions)
	    net = lays.conv2d(inputs, 128, [5, 5], stride=3, padding='SAME')
	    net = lays.conv2d(net, 64, [5, 5], stride=2, padding='SAME')
	    net = lays.conv2d(net, 32, [5, 5], stride=1, padding='SAME')
	    
	    # Decoder (tranposed convolutions)
	    net = lays.conv2d_transpose(net, 64, [5, 5], stride=1, padding='SAME')
	    net = lays.conv2d_transpose(net, 128, [5, 5], stride=2, padding='SAME')
	    net = lays.conv2d_transpose(net, channels, [5, 5], stride=3, padding='SAME', activation_fn=tf.nn.tanh)

    return net
