import numpy as np
from PIL import Image
import tensorflow as tf
from vgg_network import VGGNetwork, load_image


def leaky_relu(input_layer, alpha):
    return tf.maximum(tf.mul(input_layer, alpha), input_layer)


def conv2d(name, input_layer, w, b):
    # TODO: Mirror pad? I'm not sure how important this is.
    conv_output = tf.nn.conv2d(input_layer, w, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.bias_add(conv_output, b)


def spatial_batch_norm(input_layer, name='spatial_batch_norm'):
    """
    Batch-normalizes the layer as in http://arxiv.org/abs/1502.03167
    This is important since it allows the different scales to talk to each other when they get joined.
    """
    mean, variance = tf.nn.moments(input_layer, [0, 1, 2])
    variance_epsilon = 0.01  # TODO: Check what this value should be
    inv = tf.rsqrt(variance + variance_epsilon)
    num_channels = input_layer.get_shape().as_list()[3] # TODO: Clean this up
    scale = tf.Variable(tf.random_uniform([num_channels]), name='scale')  # TODO: How should these initialize?
    offset = tf.Variable(tf.random_uniform([num_channels]), name='offset')
    return_val = tf.sub(tf.mul(tf.mul(scale, inv), tf.sub(input_layer, mean)), offset, name=name)
    return return_val

def input_pyramid(name, M, batch_size, k=5):
    """
    Generates k inputs at different scales, with MxM being the largest.
    """
    with tf.get_default_graph().name_scope(name):
        return_val = [tf.placeholder(tf.float32, [batch_size, M//(2**x), M//(2**x), 3], name=str(x)) for x in range(k)]
        return_val.reverse()
    return return_val


def noise_pyramid(M, batch_size, k=5):
    return [np.random.rand(batch_size, M//(2**x), M//(2**x), 3) for x in range(k)][::-1]


def conv_block(name, input_layer, kernel_size, out_channels):
    """
    Per Ulyanov et el, this is a block consisting of
        - Mirror pad (TODO)
        - Number of maps from a convolutional layer equal to out_channels (multiples of 8)
        - Spatial BatchNorm
        - LeakyReLu
    """
    with tf.get_default_graph().name_scope(name):
        in_channels = input_layer.get_shape().as_list()[-1]

        # Xavier initialization, http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        # The application of this method here seems unorthodox since we're using ReLU, not sigmoid or tanh.
        low = -np.sqrt(6.0/(in_channels + out_channels))
        high = np.sqrt(6.0/(in_channels + out_channels))
        weights = tf.Variable(tf.random_uniform([kernel_size, kernel_size, in_channels, out_channels], minval=low, maxval=high), name='weights')
        biases = tf.Variable(tf.random_uniform([out_channels], minval=low, maxval=high), name='biases')
        conv = conv2d('conv', input_layer, weights, biases)
        batch_norm = spatial_batch_norm(conv)
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
        upsampled = tf.image.resize_nearest_neighbor(lower_res_layer, higher_res_layer.get_shape().as_list()[1:3])
        batch_norm_lower = spatial_batch_norm(upsampled, 'normLower')
        batch_norm_higher = spatial_batch_norm(higher_res_layer, 'normHigher')
    return tf.concat(3, [batch_norm_lower, batch_norm_higher])


class TextureNetwork(object):
    inputDimension = 224
    channelStepSize = 8
    batchSize = 1  # 16 in the paper
    epochs = 1000  # 2000 in the paper

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            """
            Construct the texture network graph structure
            """
            noise_inputs = input_pyramid("noise", self.inputDimension, self.batchSize)
            current_channels = 8
            current_noise_aggregate = noise_inputs[0]
            for noise_frame in noise_inputs[1:]:
                low_res_out = conv_chain("chain_lower_%d" % current_channels, current_noise_aggregate, current_channels)
                high_res_out = conv_chain("chain_higher", noise_frame, self.channelStepSize)
                current_channels += self.channelStepSize
                current_noise_aggregate = join_block("join_%d" % (current_channels + self.channelStepSize), low_res_out, high_res_out)
            final_chain = conv_chain("output_chain", current_noise_aggregate, current_channels)
            output = conv_block("output", final_chain, kernel_size=1, out_channels=3)

            """
            Calculate style loss by computing gramians from both the output of the texture net above and from the
            texture sample image.
            """
            texture_vgg = VGGNetwork("texture_vgg", output)
            texture_sample_image = tf.placeholder("float", [1, 224, 224, 3])
            image_vgg = VGGNetwork("image_vgg", texture_sample_image)

            # The tiling here is necessary because we're operating on a batch of noise samples, but only a one texture
            layers = [i for i in range(1, 6)]
            gramian_diffs = [tf.sub(texture_vgg.gramian_for_layer(x), tf.tile(image_vgg.gramian_for_layer(x), [self.batchSize, 1, 1])) for x in layers]
            Ns = [texture_vgg.channels_for_layer(i) for i in layers]
            Ms = [g.get_shape().as_list()[1] * g.get_shape().as_list()[2] for g in gramian_diffs]
            scaled_diffs = [tf.div(tf.square(g), 4*(N**2)*(M**2)) for g, N, M in zip(gramian_diffs, Ns, Ms)]
            # 1 / len(layers) = w_l
            loss = tf.div(tf.add_n([tf.reduce_sum(x) for x in scaled_diffs]), len(layers))
            optimizer = tf.train.GradientDescentOptimizer(0.1)
            train_step = optimizer.minimize(loss)

            """
            Train over epochs, printing loss at each one
            """
            texture_image = load_image("img/img.jpg").reshape((1, 224, 224, 3))
            saver = tf.train.Saver()
            with tf.Session() as sess:
                init = tf.initialize_all_variables()
                sess.run(init)

                for i in range(self.epochs):
                    feed_dict = {}
                    noise = noise_pyramid(self.inputDimension, self.batchSize)
                    for index, noise_frame in enumerate(noise_inputs):
                        feed_dict[noise_frame] = noise[index]
                    feed_dict[texture_sample_image] = texture_image
                    train_step.run(feed_dict=feed_dict)
                    if i > 0 and i % 20 == 0:  # TODO: Make this interval an argument
                        saver.save(sess, "models/snapshot-%d.ckpt" % i)
                        network_out = sess.run([output], feed_dict=feed_dict)
                        img = Image.fromarray(network_out[0][0, :, :, :], "RGB")
                        img.save("img/iteration-%d.jpeg" % i)


# TODO: Add argument parsing and command-line args
TextureNetwork()
