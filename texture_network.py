import numpy as np
import tensorflow as tf


def leaky_relu(input_layer, alpha):
    return tf.maximum(alpha * input_layer, input_layer)


def conv2d(name, input_layer, w, b):
    # TODO: Mirror pad? I'm not sure how important this is.
    conv_output = tf.nn.conv2d(input_layer, w, strides=[1, 1, 1, 1], padding='SAME')
    conv_with_bias = tf.nn.bias_add(conv_output, b)
    return tf.nn.relu(conv_with_bias, name=name)


def norm(name, input_layer, lsize=4):
    # TODO: Use batch normalization instead of local resoponse norm.
    # This is important since it allows the different scales to talk to each other.
    return tf.nn.local_response_normalization(input_layer, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def input_pyramid(name, M, k=5):
    """
    Generates k inputs at different scales, with MxM being the largest.
    """
    with tf.get_default_graph().name_scope(name):
        return_val = [tf.placeholder(tf.float32, [1, M//(2**x), M//(2**x), 3], name=str(x)) for x in range(k)][::-1]
    return return_val


def noise_pyramid(M, k=5):
    return [np.random.rand(1, M//(2**x), M//(2**x), 3) for x in range(k)][::-1]


def conv_block(name, input_layer, kernel_size, out_channels):
    """
    Per Ulyanov et el, this is a block consisting of
        - Mirror pad (TODO)
        - Number of maps from a convolutional layer equal to out_channels (multiples of 8)
        - BatchNorm (TODO)
        - LeakyReLu
    """
    with tf.get_default_graph().name_scope(name):
        in_channels = input_layer.get_shape().as_list()[-1]
        weights = tf.Variable(tf.random_normal([kernel_size, kernel_size, in_channels, out_channels]), name='weights')
        biases = tf.Variable(tf.random_normal([out_channels]), name='biases')
        conv = conv2d('conv', input_layer, weights, biases)
        batch_norm = norm('norm', conv, lsize=4)
        relu = leaky_relu(batch_norm, .01)
        return relu


def conv_chain(name, input_layer, out_channels):
    """
    A sequence of three conv_block units with 3x3, 3x3, and 1x1 kernels respectively.
    There's nothing inherently magical about this abstraction, but it's a repeated pattern in the Ulyanov et el network.
    """
    with tf.get_default_graph().name_scope(name):
        block1 = conv_block("layer1", input_layer, kernel_size=3, out_channels=out_channels)
        block2 = conv_block("layer1", block1, kernel_size=3, out_channels=out_channels)
        block3 = conv_block("layer1", block2, kernel_size=1, out_channels=out_channels)
    return block3


def join_block(name, lower_res_layer, higher_res_layer):
    """
    A block that combines two resolutions by upsampling the lower, batchnorming both, and concatting.
    """
    with tf.get_default_graph().name_scope(name):
        # TODO: Decide what kind of upsampling to use
        upsampled = tf.image.resize_bilinear(lower_res_layer, higher_res_layer.get_shape().as_list()[1:3])
        batch_norm_lower = norm('normLower', upsampled, lsize=4)
        batch_norm_higher = norm('normHigher', higher_res_layer, lsize=4)
    return tf.concat(3, [batch_norm_lower, batch_norm_higher])


class TextureNetwork(object):
    inputDimension = 224
    channelStepSize = 8

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            noise_inputs = input_pyramid("noise", self.inputDimension)
            current_channels = 0
            current_noise_aggregate = noise_inputs[0]
            for noise_frame in noise_inputs[1:]:
                current_channels += self.channelStepSize
                low_res_out = conv_chain("chain_lower_%d" % current_channels, current_noise_aggregate, current_channels)
                high_res_out = conv_chain("chain_higher", noise_frame, self.channelStepSize)
                current_noise_aggregate = join_block("join_%d" % (current_channels + self.channelStepSize), low_res_out, high_res_out)
            final_chain = conv_chain("output_chain", current_noise_aggregate, current_channels)
            output = conv_block("output", final_chain, kernel_size=1, out_channels=current_channels)

            with tf.Session() as sess:
                init = tf.initialize_all_variables()
                sess.run(init)
                # Log the graph structure for inspection in Tensorboard
                writer = tf.train.SummaryWriter("tensorboard-log", max_queue=10, flush_secs=1)
                writer.add_graph(self.graph.as_graph_def())
                writer.flush()
                # Test running noise samples through the network
                feed_dict = {}
                noise = noise_pyramid(self.inputDimension)
                for index, noise_frame in enumerate(noise_inputs):
                    feed_dict[noise_frame] = noise[index]
                print(sess.run(output, feed_dict=feed_dict))


# TODO: Add argument parsing and command-line args
TextureNetwork()
