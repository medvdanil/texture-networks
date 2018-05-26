"""
Loads vgg16 from disk as a tensorflow model with batching to process style, content, and synthesized images
simultaneously (while abstracting the accessors to compute style/content loss based on that representation).
"""
from style_helpers import gramian
import tensorflow as tf


class VGGNetwork(object):

    def __init__(self, name, input, i, j, k):
        """
        :param input: A 4D-tensor of shape [batchSize, 224, 224, 3]
                [0:i, :, :, :] holds i style images,
                [i:i+j, :, :, :] holds j content images,
                [i+j:i+j+k, :, :, :] holds k synthesized images
        """
        self.name = name
        self.num_style = i
        self.num_content = j
        self.num_synthesized = k
        with open("data/vgg16-20160129.tfmodel", mode='rb') as f:
            file_content = f.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file_content)
        tf.import_graph_def(graph_def, input_map={"images": input}, name=self.name)

    def print_op_names(self):
        """
        Utility for inspecting graph layers since this model is a bit big for Tensorboard.
        """
        print([op.name for op in tf.get_default_graph().get_operations()])

    def channels_for_layer(self, layer):
        activations = tf.get_default_graph().get_tensor_by_name("%s/conv%d_1/Relu:0" % (self.name, layer))
        return activations.get_shape().as_list()[3]

    def gramian_for_layer(self, layer):
        """
        Returns a matrix of cross-correlations between the activations of convolutional channels in a given layer.
        """
        activations = self.activations_for_layer(layer)

        # Reshape from (batch, width, height, channels) to (batch, channels, width, height)
        shuffled_activations = tf.transpose(activations, perm=[0, 3, 1, 2])
        return gramian(shuffled_activations)

    def activations_for_layer(self, layer):
        """
        :param layer: A tuple that indexes into the convolutional blocks of the VGG Net
        """
        return tf.get_default_graph().get_tensor_by_name("{0}/conv{1}_{2}/Relu:0".format(self.name, layer[0], layer[1]))

    def style_loss(self, layers):
        activations = [self.activations_for_layer(i) for i in layers]
        gramians = [self.gramian_for_layer(x) for x in layers]
        # Slices are for style and synth image
        gramian_diffs = [
            tf.subtract(
                tf.tile(tf.slice(g, [0, 0, 0], [self.num_style, -1, -1]), [self.num_synthesized - self.num_style + 1, 1, 1]),
                tf.slice(g, [self.num_style + self.num_content, 0, 0], [self.num_synthesized, -1, -1]))
            for g in gramians]
        Ns = [g.get_shape().as_list()[2] for g in gramians]
        Ms = [a.get_shape().as_list()[1] * a.get_shape().as_list()[2] for a in activations]
        scaled_diffs = [tf.square(g) for g in gramian_diffs]
        style_loss = tf.div(
            tf.add_n([tf.div(tf.reduce_sum(x), 4 * (N ** 2) * (M ** 2)) for x, N, M in zip(scaled_diffs, Ns, Ms)]),
            len(layers))
        return style_loss
