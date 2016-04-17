import tensorflow as tf


def gramian(activations):
    # Takes (batches, channels, width, height) and computes gramians of dimension (batches, channels, channels)
    activations_shape = activations.get_shape().as_list()
    """
    Instead of iterating over #channels width by height matrices and computing similarity, we vectorize and compute
    the entire gramian in a single matrix multiplication.
    """
    vectorized_activations = tf.reshape(activations,
                                        [activations_shape[0], activations_shape[1], -1])
    transposed_vectorized_activations = tf.transpose(vectorized_activations, perm=[0, 2, 1])
    mult = tf.batch_matmul(vectorized_activations, transposed_vectorized_activations)
    return mult


def total_variation(image_batch):
    """
    :param image_batch: A 4D tensor of shape [batch_size, width, height, channels]
    """
    batch_shape = image_batch.get_shape().as_list()
    width = batch_shape[1]
    left = tf.slice(image_batch, [0, 0, 0, 0], [-1, width - 1, -1, -1])
    right = tf.slice(image_batch, [0, 1, 0, 0], [-1, -1, -1, -1])

    height = batch_shape[2]
    top = tf.slice(image_batch, [0, 0, 0, 0], [-1, -1, height - 1, -1])
    bottom = tf.slice(image_batch, [0, 0, 1, 0], [-1, -1, -1, -1])

    # left and right are 1 less wide than the original, top and bottom 1 less tall
    # In order to combine them, we take 1 off the height of left-right, and 1 off width of top-bottom
    horizontal_diff = tf.slice(tf.sub(left, right), [0, 0, 0, 0], [-1, -1, height - 1, -1])
    vertical_diff = tf.slice(tf.sub(top, bottom), [0, 0, 0, 0], [-1, width - 1, -1, -1])

    sum_of_pixel_diffs_squared = tf.add(tf.square(horizontal_diff), tf.square(vertical_diff))
    total_variation = tf.reduce_sum(tf.sqrt(sum_of_pixel_diffs_squared))
    # TODO: Should this be normalized by the number of pixels?
    return total_variation
