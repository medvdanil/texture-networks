import numpy as np
from os import listdir
from os.path import isfile, join

from network_helpers import load_image

class COCODataBatcher(object):

    def __init__(self, path):
        self.path = path
        all_files = listdir(self.path)
        self.files = [f for f in all_files if isfile(join(self.path, f))]
        self.current_index = 0
        self.num_files = len(self.files)

    def get_batch(self, batch_size):
        assert batch_size < self.num_files, 'Batch size bigger than file count'
        if self.current_index + batch_size > self.num_files:
            self.current_index = 0
        file_batch = self.files[self.current_index:self.current_index + batch_size]
        self.current_index += batch_size
        return np.array([load_image(join(self.path, filename)) for filename in file_batch])
