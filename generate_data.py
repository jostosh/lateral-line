import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import six.moves.cPickle as pickle
from tqdm import trange

from common.plot3d import plot3d
from latline.experiment_config import DataConfig, parse_config_args
from latline.latline import Latline


def generate_data(cfg, mode='train'):
    """
    This function generates the simulated sensor data given the following parameters:

    Paramaters:
        :param N_examples:      The number of examples to be generated
        :param x_range:         The range along x-axis
        :param y_range:         The range along y-axis
        :param z_range:         The range along z-axis
        :param v:               The speed of the sphere
        :param d_theta_range:   The range of the delta angle
        :param resolution:      The resolution along for the target matrix (i.e. what the depth should be of the output)
        :param sigma:           The sigma parameter used in the similarity function
        :param N_sensors:       The number of sensors per sensor array
        :param sensor_range:    The range of the sensor array
        :param display:         Whether to display the simulation in 3D
        :param tau:             The parameter defining the time-window for multiple frames of excitation as input
        :param mode:            Whether it is train or test data. Used for the progress bar.

    Return:
        :return Two multi-dimensional arrays containing the train data and the output targets, respectively
    """
    N_examples = cfg.N_train if mode == 'train' else cfg.N_test

    # pre-alloc data and labels
    data    = np.zeros((N_examples, 2, cfg.N_sensors, cfg.tau))
    targets  = np.zeros((N_examples, 2, cfg.N_sensors, cfg.resolution))

    # s should be interpreted as in Boulogne et al (2015)
    s = np.linspace(cfg.sensor_range[0], cfg.sensor_range[1], cfg.N_sensors)
    x_grid = s
    y_grid = s
    z_grid = np.linspace(cfg.z_range[0], cfg.z_range[1], cfg.resolution)

    # xx and yy are matrices for evaluating a Gaussian that is centered at a sphere
    # xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)

    x_mesh2d, target_mesh = np.meshgrid(x_grid, z_grid)

    x_mesh3d, y_mesh3d, z_mesh3d = np.meshgrid(x_grid, y_grid, z_grid)

    # Init lateral line for train data
    latline = Latline(cfg)

    if cfg.display:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # We initialize buffers for the two sensor arrays that contain the excitations over multiple timesteps
    buffer0 = deque()
    buffer1 = deque()

    # Loop through all data points
    t = trange(N_examples + cfg.tau)
    t.set_description("Generating {} data".format(mode))
    for i in t:
        # Obtain x and y locations of all spheres and the corresponding sensor measurements
        xs, ys, zs, fluid_v0, fluid_v1 = latline.step()  # xs' rows correspond to the spheres

        # Compute each Gaussian separately
        target0 = np.asarray([np.exp(-((x_mesh2d - xi) ** 2 + (target_mesh - np.sqrt((yi+0.5)**2 + zi**2)) ** 2)
                                     / (2 * cfg.sigma**2))
                              for (xi, yi, zi) in zip(xs, ys, zs)])
        target1 = np.asarray([np.exp(-((x_mesh2d - xi) ** 2 + (target_mesh - np.sqrt((yi-0.5)**2 + zi**2)) ** 2)
                                     / (2 * cfg.sigma**2))
                              for (xi, yi, zi) in zip(xs, ys, zs)])
        # And then take the maximum over all spheres.
        target0 = np.max(target0, axis=0)
        target1 = np.max(target1, axis=0)

        # Save measurements and heatmap
        buffer0.append(fluid_v0 * cfg.sensitivity)
        buffer1.append(fluid_v1 * cfg.sensitivity)

        if i >= cfg.tau:
            buffer0.popleft()
            buffer1.popleft()

            data[i - cfg.tau][0] = np.transpose(buffer0)
            data[i - cfg.tau][1] = np.transpose(buffer1)
            targets[i - cfg.tau][0] = np.transpose(target0.reshape((cfg.resolution, cfg.N_sensors)))
            targets[i - cfg.tau][1] = np.transpose(target1.reshape((cfg.resolution, cfg.N_sensors)))

        if cfg.display and i < 100:
            # Plot the situation in 3D
            plot3d(cfg, ax, fluid_v0, fluid_v1, xs, ys, zs, x_mesh3d, y_mesh3d, z_mesh3d, latline, s, target0, target1)

    return data, targets


if __name__ == "__main__":
    """
    This script generates the data to be used for the training of the CNNs in train.py
    """
    config = DataConfig(parse_config_args(mode='data'))

    # Obtain data for train, test and validation sets
    train_data, train_labels = generate_data(config, mode='train')
    test_data, test_labels = generate_data(config, mode='test')
    project_folder = os.path.dirname(os.path.realpath(__file__))

    # Serialize the data
    os.makedirs(os.path.join(project_folder, 'data'), exist_ok=True)
    with open(os.path.join(project_folder, 'data', 'multisphere_parallel.pickle'), 'wb') as f:
        pickle.dump([train_data, train_labels, test_data, test_labels], f)
