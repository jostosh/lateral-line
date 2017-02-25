import os
import matplotlib.pyplot as plt
import numpy as np
import six.moves.cPickle as pickle
from latline.experiment_config import DataConfig, parse_config_args
from latline.latline import Latline
from plot3d import plot3d
from tqdm import trange
from collections import deque


def generate_data(N_examples, x_range, y_range, z_range, v, d_theta_range, resolution, sigma, N_sensors, sensor_range,
                  display, tau=2, mode='train'):
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
    # pre-alloc data and labels
    data    = np.zeros((N_examples, 2, N_sensors, tau))
    targets  = np.zeros((N_examples, 2, N_sensors, resolution))

    # s should be interpreted as in Boulogne et al (2015)
    s = np.linspace(sensor_range[0], sensor_range[1], N_sensors)
    x_grid = s
    y_grid = s
    z_grid = np.linspace(z_range[0], z_range[1], resolution)

    # xx and yy are matrices for evaluating a Gaussian that is centered at a sphere
    # xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)

    x_mesh2d, target_mesh = np.meshgrid(x_grid, z_grid)

    x_mesh3d, y_mesh3d, z_mesh3d = np.meshgrid(x_grid, y_grid, z_grid)

    # Init lateral line for train data
    latline = Latline(x_range=x_range, y_range=y_range, z_range=z_range, d_theta_range=d_theta_range,
                      sensor_range=sensor_range, n_sensors=N_sensors, min_spheres=1, max_spheres=2, v=v)

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # We initialize buffers for the two sensor arrays that contain the excitations over multiple timesteps
    buffer0 = deque()
    buffer1 = deque()

    # Loop through all data points
    t = trange(N_examples + tau)
    t.set_description("Generating {} data".format(mode))
    for i in t:
        # Obtain x and y locations of all spheres and the corresponding sensor measurements
        xs, ys, zs, fluid_v_x, fluid_v_y = latline.step()  # xs' rows correspond to the spheres

        # Compute each Gaussian separately
        target0 = np.asarray([np.exp(-((x_mesh2d - xi) ** 2 + (target_mesh - np.sqrt((yi+0.5)**2 + zi**2)) ** 2)
                                     / (2 * sigma**2))
                              for (xi, yi, zi) in zip(xs, ys, zs)])
        target1 = np.asarray([np.exp(-((x_mesh2d - xi) ** 2 + (target_mesh - np.sqrt((yi-0.5)**2 + zi**2)) ** 2)
                                     / (2 * sigma**2))
                              for (xi, yi, zi) in zip(xs, ys, zs)])
        # And then take the maximum over all spheres.
        target0 = np.max(target0, axis=0)
        target1 = np.max(target1, axis=0)

        # Save measurements and heatmap
        buffer0.append(fluid_v_x)
        buffer1.append(fluid_v_y)

        if i >= tau:
            buffer0.popleft()
            buffer1.popleft()

            data[i - tau][0] = np.transpose(buffer0)
            data[i - tau][1] = np.transpose(buffer1)
            targets[i - tau][0] = np.transpose(target0.reshape((resolution, N_sensors)))
            targets[i - tau][1] = np.transpose(target1.reshape((resolution, N_sensors)))

        if display:
            # Plot the situation in 3D
            plot3d(N_sensors, ax, fluid_v_x, fluid_v_y, latline, s, sensor_range, target0, target1, x_mesh3d, xs,
                   y_mesh3d, ys, z_mesh3d, z_range, resolution, zs)

    return data, targets


if __name__ == "__main__":
    """
    This script generates the data to be used for the training of the CNNs in train.py
    """
    config = DataConfig(parse_config_args(mode='data'))

    # Obtain data for train, test and validation sets
    train_data, train_labels = generate_data(N_examples=config.N_train, x_range=config.x_range, y_range=config.y_range,
                                             z_range=config.z_range, v=config.v, d_theta_range=config.d_theta_range,
                                             resolution=config.z_res, sigma=config.sigma, N_sensors=config.N_sensors,
                                             sensor_range=config.sensor_range, display=config.display,
                                             tau=config.tau, mode='train')
    test_data, test_labels = generate_data(N_examples=config.N_test, x_range=config.x_range, y_range=config.y_range,
                                           z_range=config.z_range, v=config.v, d_theta_range=config.d_theta_range,
                                           resolution=config.z_res, sigma=config.sigma, N_sensors=config.N_sensors,
                                           sensor_range=config.sensor_range, display=config.display,
                                           tau=config.tau, mode='test')
    project_folder = os.path.dirname(os.path.realpath(__file__))

    # Serialize the data
    os.makedirs(os.path.join(project_folder, 'data'), exist_ok=True)
    with open(os.path.join(project_folder, 'data', 'multisphere_parallel.pickle'), 'wb') as f:
        pickle.dump([train_data, train_labels, test_data, test_labels], f)
