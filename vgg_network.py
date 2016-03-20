"""
Partially taken from https://github.com/ry/tensorflow-vgg16/blob/master/tf_forward.py
Loads vgg16 from disk as a tensorflow model.
"""
import skimage.io
import skimage.transform
import tensorflow as tf


def load_image(path):
    # load image
    img = skimage.io.imread(path) / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


class VGGNetwork(object):

    def __init__(self, name, input):
        self.name = name
        with open("models/vgg16.tfmodel", mode='rb') as f:
            fileContent = f.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)
        tf.import_graph_def(graph_def, input_map={ "images": input }, name=self.name)

    def print_op_names(self):
        """
        Utility for inspecting graph layers since this model is a bit big for Tensorboard.
        """
        print([op.name for op in tf.get_default_graph().get_operations()])

    # TODO: Write a test asserting this function behaves as expected. Validating shapes would be nice.
    def gramian_for_layer(self, layer):
        """
        Returns a matrix of cross-correlations between the activations of convolutional channels in a given layer.
        """
        activations = tf.get_default_graph().get_tensor_by_name("%s/conv%d_1/Conv2D:0" % (self.name, layer))

        # Reshape from (batch, width, height, channels) to (batch, channels, width, height)
        filter_activations = tf.transpose(activations, perm=[0, 3, 1, 2])
        activations_shape = filter_activations.get_shape().as_list()

        # Vectorize the width*height activations
        vectorized_activations = tf.reshape(filter_activations,
                                            [activations_shape[0], activations_shape[1], -1])
        transposed_vectorized_activations = tf.transpose(vectorized_activations, perm=[0, 2, 1])

        # Calculate a (batch, channels, channels) sized stack of gramians
        mult = tf.batch_matmul(vectorized_activations, transposed_vectorized_activations)
        return mult