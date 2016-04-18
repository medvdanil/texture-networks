"""
Implementation of style transfer network defined in "Perceptual Losses for Real-Time
Style Transfer and Super-Resolution", http://arxiv.org/abs/1603.08155
"""

import tensorflow as tf

from coco_data import COCODataBatcher
from network_helpers import conv2d_with_weights, conv2d_block_with_weights, spatial_batch_norm, load_image
from style_helpers import total_variation
from vgg_network import VGGNetwork


def nonresidual_block(input_layer, num_filters):
    """
    Constructs a block without residual
    """
    with tf.get_default_graph().name_scope('nonresidual-block'):
        conv1 = conv2d_with_weights(input_layer, 3, num_filters)
        bn1 = spatial_batch_norm(conv1)
        relu = tf.nn.relu(bn1)
        conv2 = conv2d_with_weights(relu, 3, num_filters)
        output = spatial_batch_norm(conv2)
    return output


def residual_block(input_layer, num_filters):
    """
    Constructs a block as detailed in http://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    """
    with tf.get_default_graph().name_scope('residual-block'):
        output = tf.add(nonresidual_block(input_layer, num_filters), input_layer)
    return output


class PerceptualLossNetwork(object):
    inputDimension = 224
    batchSize = 4

    def __init__(self, style_image_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            inputDim = self.inputDimension
            batchSize = self.batchSize
            halfInputDim = int(inputDim / 2)
            quarterInputDim = int(inputDim / 4)
            self.content_image = tf.placeholder(tf.float32, [batchSize, inputDim, inputDim, 3])
            layer1 = conv2d_block_with_weights(self.content_image, 9, 32)
            layer2 = conv2d_block_with_weights(layer1, 3, 64, stride=2)
            layer3 = conv2d_block_with_weights(layer2, 3, 128, stride=2)
            layer4 = residual_block(layer3, 128)
            layer5 = residual_block(layer4, 128)
            layer6 = residual_block(layer5, 128)
            layer7 = residual_block(layer6, 128)
            layer8 = residual_block(layer7, 128)
            layer9 = tf.nn.conv2d_transpose(layer8, tf.Variable(tf.random_uniform([quarterInputDim, quarterInputDim, 64, 128])), [batchSize, halfInputDim, halfInputDim, 64], [1, 2, 2, 1])
            layer10 = tf.nn.conv2d_transpose(layer9, tf.Variable(tf.random_uniform([halfInputDim, halfInputDim, 32, 64])), [batchSize, inputDim, inputDim, 32], [1, 2, 2, 1])
            self.output = conv2d_block_with_weights(layer10, 3, 3)
            self.style_image = tf.to_float(tf.reshape(tf.constant(load_image(style_image_path)), [1, 224, 224, 3]))

            vgg_input = tf.concat(0, [self.style_image, self.content_image, self.output])
            # Feed output to VGG, compute loss against input image and style image.
            vgg_net = VGGNetwork('vgg', vgg_input, 1, self.batchSize, self.batchSize)
            self.style_loss = vgg_net.style_loss([(1, 2), (2, 2), (3, 3), (4, 3)])
            self.content_loss = vgg_net.content_loss([(2, 2)])
            # Alternatively, self.network_loss = vgg_net.combined_loss([(1, 2), (2, 2), (3, 3), (4, 3)], [(2, 2)])
            # but this is currently more useful for debugging and inspecting the losses separately
            self.network_loss = tf.add(self.style_loss, self.content_loss)
            self.variation_regularizer_loss = total_variation(self.output)  # For smoothing
            self.loss = tf.add(self.network_loss, self.variation_regularizer_loss)

    def run_train(self, epochs):
        with self.graph.as_default():
            # Gradient descent. Draw minibatches of images from COCO.
            batcher = COCODataBatcher('data')
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                               use_locking=False, name='Adam')
            train_step = optimizer.minimize(self.loss)

            with tf.Session() as sess:
                init = tf.initialize_all_variables()
                sess.run(init)
                for e in range(epochs):
                    batch = batcher.get_batch(self.batchSize)
                    train_step.run(feed_dict={self.content_image: batch})
                    print("loss after ", e, sess.run(self.loss, feed_dict={self.content_image: batch}))
