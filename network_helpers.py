import numpy as np
import skimage.io
import skimage.transform
import tensorflow as tf

images_shape = 448, 448

def load_image(path):
    """
    Taken from https://github.com/ry/tensorflow-vgg16/blob/master/tf_forward.py
    """
    # load image
    img = skimage.io.imread(path) / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    # resize to images_shape
    resized_img = skimage.transform.resize(crop_img, images_shape)
    return resized_img

def conv2d_block_with_weights(input_layer, kernel_size, num_filters, stride=1):
    # TODO: Consolidate this with conv_block from texture_network
    conv = conv2d_with_weights(input_layer, kernel_size, num_filters, stride=stride)
    bn = spatial_batch_norm(conv)
    return tf.nn.relu(bn)

def conv2d_with_weights(input_layer, kernel_size, num_filters, name="conv2d", stride=1):
    in_channels = input_layer.get_shape().as_list()[-1] # This assumes a certain ordering :/
    # Xavier initialization, http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    # TODO: Maybe make weight initialization method an option.
    low = -np.sqrt(6.0 / (in_channels + num_filters))
    high = np.sqrt(6.0 / (in_channels + num_filters))
    weights = tf.Variable(
        tf.random_uniform([kernel_size, kernel_size, in_channels, num_filters], minval=low, maxval=high),
        name=name+'-weights')
    biases = tf.Variable(tf.random_uniform([num_filters], minval=low, maxval=high), name=name+'-biases')
    conv_output = tf.nn.conv2d(input_layer, weights, strides=[1, stride, stride, 1], padding='SAME')
    return tf.nn.bias_add(conv_output, biases)


def conv2d(input_layer, w, b, stride=1):
    conv_output = tf.nn.conv2d(input_layer, w, strides=[1, stride, stride, 1], padding='SAME')
    return tf.nn.bias_add(conv_output, b)


def spatial_batch_norm(input_layer, name='spatial_batch_norm'):
    """
    Batch-normalizes the layer as in http://arxiv.org/abs/1502.03167
    This is important since it allows the different scales to talk to each other when they get joined.
    """
    mean, variance = tf.nn.moments(input_layer, [0, 1, 2])
    variance_epsilon = 0.01  # TODO: Check what this value should be
    inv = tf.rsqrt(variance + variance_epsilon)
    num_channels = input_layer.get_shape().as_list()[3]  # TODO: Clean this up
    scale = tf.Variable(tf.random_uniform([num_channels]), name='scale')  # TODO: How should these initialize?
    offset = tf.Variable(tf.random_uniform([num_channels]), name='offset')
    return_val = tf.subtract(tf.multiply(tf.multiply(scale, inv), tf.subtract(input_layer, mean)), offset, name=name)
    return return_val
