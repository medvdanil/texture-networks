"""
Partially taken from https://github.com/ry/tensorflow-vgg16/blob/master/tf_forward.py
Loads vgg16 from disk as a tensorflow model.
"""
import tensorflow as tf

import skimage.io
import skimage.transform
def load_image(path):
  # load image
  img = skimage.io.imread(path)
  img = img / 255.0
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

    def __init__(self):
        with open("models/vgg16.tfmodel", mode='rb') as f:
            fileContent = f.read()

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)
        images = tf.placeholder("float", [None, 224, 224, 3])

        tf.import_graph_def(graph_def, input_map={ "images": images })

        self.graph = tf.get_default_graph()
        prob_tensor = self.graph.get_tensor_by_name("import/prob:0")
        test_image = load_image("img/img.jpg")
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            batch = test_image.reshape((1, 224, 224, 3))
            feed_dict = { images: batch }
            print("running")
            # Test that the network runs.
            prob = sess.run(prob_tensor, feed_dict={images: batch})
            print(prob)

    def printOpNames(self):
        print([op.name for op in self.graph.get_operations()])


q = VGGNetwork()
q.printOpNames()
