import tensorflow as tf
import input_data
import glob
import os

# Parameters
channels = 1

dataset_path = 'data/40nicer'
image_width = 270
image_height = 90
batch_size = 30
image_data = input_data.load_images(dataset_path, image_width, image_height)

def summary_layer(net, name):
    tf.summary.image(name, tf.expand_dims(tf.transpose(net, [3, 0, 1, 2])[0], 3), max_outputs=1)
    print(net.shape)

def conv_layer_dropout(net, layer, out_channels, filter_dims, strides, padding, name, dropout, act_f = tf.nn.relu):
    net = tf.layers.dropout(net, dropout)
    return conv_layer(net, layer, out_channels, filter_dims, strides, padding, name, act_f)

def conv_layer(net, layer, out_channels, filter_dims, strides, padding, name, act_f = tf.nn.relu):
    net = layer(net, out_channels, filter_dims, strides=strides, padding=padding)
    net = act_f(net)
    summary_layer(net, name)
    return net


def pre_train_conv_layer(epochs, inputs, layer, out_channels, filt, strides, name, act_f = tf.nn.relu):
    forward = layer
    backward = tf.layers.conv2d_transpose if layer is tf.layers.conv2d else tf.layers.conv2d

    name_scope = name + '_pretrain_scope'
    with tf.variable_scope(name_scope) as vs:
        net = conv_layer(inputs, forward, out_channels, filt, strides, 'SAME', name)
        out = conv_layer(net, backward, int(inputs.shape[-1]), filt, strides, 'SAME', name + '_pretrain')

    trainable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
    var_list = [v for v in trainable_variables if name in v.name]
    if len(var_list) != 4:
        raise Exception("No two unique output layers to pretrain")

    cost = tf.reduce_mean(tf.square(out - inputs))
    step = tf.train.GradientDescentOptimizer(1e-3).minimize(cost, var_list=var_list)

    return net, step, cost

files = glob.glob('./pretrain/*')
for f in files:
    os.remove(f)
writer = tf.summary.FileWriter("./pretrain", graph=tf.get_default_graph())
def pretrain(epochs, step, loss, placeholder, name):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_op = tf.summary.merge_all()

    for i in range(epochs):
        input_, blurred = image_data.next_batch(batch_size)
        _, cost, summary = sess.run([step, loss, summary_op], feed_dict={placeholder: input_})
        writer.add_summary(summary, i)
        print(i, "Pretrain " + name, cost)


pretrain_steps = 5
def autoencoder(original, inputs, batch_size, dropout=0.35):
    # Encoder
    # net =      conv_layer(inputs, tf.layers.conv2d, 256, [5, 5], (3, 3), 'SAME', 'conv1')
    net, step, loss = pre_train_conv_layer(pretrain_steps, inputs, tf.layers.conv2d, 256, [3, 3], (3, 3), 'conv1')
    pretrain(pretrain_steps, step, loss, original, 'conv1')
    # net = conv_layer_dropout(net, tf.layers.conv2d, 128, [5, 5], (2, 2), 'SAME', 'conv2', dropout)
    net, step, loss = pre_train_conv_layer(pretrain_steps, net, tf.layers.conv2d, 128, [3, 3], (2, 2), 'conv2')
    pretrain(pretrain_steps, step, loss, original, 'conv2')

    net, step, loss  = pre_train_conv_layer(pretrain_steps, net, tf.layers.conv2d, 64, [3, 3], (1, 1), 'conv3')
    pretrain(pretrain_steps, step, loss, original, 'conv2')

    # Decoder
    # net = conv_layer_dropout(net, tf.layers.conv2d_transpose, 64 , [5, 5], (1, 1), 'SAME', 'deconv1', dropout)
    net, step, loss  = pre_train_conv_layer(pretrain_steps, net, tf.layers.conv2d_transpose, 64, [3, 3], (1, 1), 'deconv1')
    # net = conv_layer_dropout(net, tf.layers.conv2d_transpose, 128, [5, 5], (2, 2), 'SAME', 'deconv2', dropout)
    net, step, loss  = pre_train_conv_layer(pretrain_steps, net, tf.layers.conv2d_transpose, 128, [3, 3], (2, 2), 'deconv2')
    # net = conv_layer_dropout(net, tf.layers.conv2d_transpose, channels, [5, 5], (3, 3), 'SAME', 'deconv3', dropout)
    net, step, loss  = pre_train_conv_layer(pretrain_steps, net, tf.layers.conv2d_transpose, channels, [3, 3], (3, 3), 'deconv3')

    # Final tanh activation
    net = tf.nn.tanh(net)
    return net
