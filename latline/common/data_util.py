import os
import pickle
import numpy as np
from tqdm import trange


class DataBatcher:
    def __init__(self, batch_size, *data_iterables):
        """
        Initializes the class from input, target and batch size.
        :param x: Data input, batch dimension assumed to be at 0 axis
        :param y: Data target, batch dimension assumed to be at 0 axis
        :param batch_size: Batch size
        """
        self._data = [np.asarray(d) for d in data_iterables]
        self.batch_size = batch_size
        self.idx = 0
        self.num_samples = len(self._data[0])
        self._shuffle()

    def next_batch(self):
        """
        Provides a simple interface for retrieving mini-batches of data
        :return: A minibatch of data
        """
        start = self.idx
        end = min(self.idx + self.batch_size, self.num_samples)
        out = [arr[start:end] for arr in self._data]

        self.idx += self.batch_size
        if self.idx >= self.num_samples:
            self.idx = 0
            self._shuffle()

        return out

    def _shuffle(self):
        """
        Shuffles the data
        """
        p = np.random.permutation(self.num_samples).astype(np.int64)
        self._data = [arr[p] for arr in self._data]

    @property
    def num_batches(self):
        return int(np.ceil(len(self._data[0]) / self.batch_size))

    def iterate_one_epoch(self):
        return trange(self.num_batches, leave=False, unit_scale=True)



def read_data(config):
    """
    Reads in the data
    """
    path = os.path.join(config.data, "{}.pickle".format(config.fnm))
    print("Reading data at {}".format(path))
    with open(path, 'rb') as f:
        return pickle.load(f)
