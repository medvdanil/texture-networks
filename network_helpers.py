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


def slice_border(t, axis, pad):
    sz = [-1, -1, -1, -1]
    sz[axis] = pad
    begin2 = [0, 0, 0, 0]
    begin2[axis] = t.get_shape().as_list()[axis] - pad
    return tf.slice(t, [0, 0, 0, 0], sz), tf.slice(t, begin2, sz)

def conv2d(input_layer, w, b, stride=1):
    print("w", w.get_shape().as_list())
    print("input_layer", input_layer.get_shape().as_list())
    tile_padding = input_layer
    for axis in 1, 2:
        b1, b2 = slice_border(tile_padding, axis, w.get_shape().as_list()[axis - 1] // 2)
        tile_padding = tf.concat([b1, tile_padding, b2], axis)
        print("tile_padding%d" % axis, tile_padding.get_shape().as_list())
    conv_output = tf.nn.conv2d(tile_padding, w, strides=[1, stride, stride, 1], padding='SAME')
    slice_beg = [0] + w.get_shape().as_list()[:2] + [0]
    slice_beg[1] //= 2; slice_beg[2] //= 2;
    slice_shp = [-1] + input_layer.get_shape().as_list()[1:3] + [-1]
    conv_crop = tf.slice(conv_output, slice_beg, slice_shp)
    return tf.nn.bias_add(conv_crop, b)


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
