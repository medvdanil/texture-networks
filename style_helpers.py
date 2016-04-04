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