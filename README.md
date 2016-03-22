## Texture Networks in Tensorflow.

The short-term goal of this project is to implement Texture Networks, as described in http://arxiv.org/abs/1603.03417. (For more background, see http://arxiv.org/abs/1508.06576) Longer term goals include exploring extensions and related architectures around the theme.

Development is being done in python 3.5, though you will need to use 2.7 to run Tensorboard.

## Setup

You will need to get a copy of the VGG16 network, I recommend the torrent listed at https://github.com/ry/tensorflow-vgg16

Set up a python 3.5 virtual environment, intall tensorflow and then `pip install -r requirements.txt`. (If that scares you, I think the only real reqs so far are tensorflow, scikit, and pillow. If you hit problems let me know.) You should then be good to run `python texture_network.py` and `python vgg_network.py` (after pointing the latter at your local vgg file, and adding an img.jpg to the img directory).
