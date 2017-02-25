import numpy as np


def distanceToIdx(d, range_d, maxi):
    idx = np.asarray((d - range_d[0]) / (range_d[1] - range_d[0]) * maxi)
    return np.minimum(np.maximum(idx, 0), maxi-1)