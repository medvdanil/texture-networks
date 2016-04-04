from numpy.testing import assert_equal
from style_helpers import gramian
import tensorflow as tf
import unittest

class TestGramians(unittest.TestCase):

    def test_gramian_size(self):
        shapes = [[10, 5, 20, 30], [1, 1, 20, 30], [10, 1, 5, 5], [1, 10, 4, 4]]
        for shape in shapes:
            batch = tf.Variable(tf.random_uniform(shape))
            result = gramian(batch)
            assert_equal(result.get_shape().as_list(), [shape[0], shape[1], shape[1]], err_msg='Gramian shape error')

    def test_gramian_value(self):
        batches = [[[[[1]], [[2]], [[3]]]],
                   [[[[0, 0]], [[1, 1]], [[2, 2]]]],
                   [[[[1, 0], [1, 0]], [[0, 2], [0, 2]], [[3, 3], [3, 3]]]]]
        results = [[[[1, 2, 3], [2, 4, 6], [3, 6, 9]]],
                   [[[0, 0, 0], [0, 2, 4], [0, 4, 8]]],
                   [[[2, 0, 6], [0, 8, 12], [6, 12, 36]]]]
        for batch, result in zip(batches, results):
            constant_batch = tf.constant(batch)
            gram = gramian(constant_batch)
            with tf.Session() as sess:
                output = sess.run(gram)
                assert_equal(output, result, err_msg='Error computing Gramian')