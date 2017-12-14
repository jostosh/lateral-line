import matplotlib.pylab as plt
import numpy as np
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size': 14})
rc('text', usetex=True)


ys = [0.2, 0.6, 0.3]
xs = [-0.5, 0.2, 0.8]

xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 32), np.linspace(0, 1, 24))
sigma = 0.025

f = np.max([np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / sigma) for x, y in zip(xs, ys)], axis=0)

print(xx.shape, f.shape)
fig = plt.figure(figsize=(5, 2.5))
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.tight_layout()
c = plt.pcolormesh(xx, yy, f)
cb = plt.colorbar(c)
cb.set_label("Density")
plt.show()