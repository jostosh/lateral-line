import os
import pickle

import numpy as np


class DataBatcher:
    def __init__(self, x, y, batch_size):
        """
        Initializes the class from input, target and batch size.
        :param x: Data input, batch dimension assumed to be at 0 axis
        :param y: Data target, batch dimension assumed to be at 0 axis
        :param batch_size: Batch size
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.idx = 0
        self._shuffle()

    def next_batch(self):
        """
        Provides a simple interface for retrieving mini-batches of data
        :return: A minibatch of data
        """
        batch_x = self.x[self.idx:min(self.idx+self.batch_size, self.x.shape[0])]
        batch_y = self.y[self.idx:min(self.idx+self.batch_size, self.y.shape[0])]
        self.idx += self.batch_size

        if self.idx >= self.x.shape[0]:
            self.idx = 0
            self._shuffle()

        return batch_x, batch_y

    def _shuffle(self):
        """
        Shuffles the data
        """
        p = np.random.permutation(self.x.shape[0])
        self.x = self.x[p]
        self.y = self.y[p]


def read_data(config):
    """
    Reads in the data
    """
    path = os.path.join(config.data, "{}.pickle".format(config.fnm))
    print("Reading data at {}".format(path))
    with open(path, 'rb') as f:
        train_x, train_y, test_x, test_y = pickle.load(f)
    return test_x, test_y, train_x, train_y