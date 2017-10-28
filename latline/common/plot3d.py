import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from matplotlib import rc
from mpl_toolkits.mplot3d import axes3d, Axes3D

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 14})
rc('text', usetex=True)


def plot3d(cfg, ax, fluid_v0, fluid_v1, xs, ys, zs, x_mesh3d, y_mesh3d, z_mesh3d, latline, s, target0, target1):
    """
    Plots the data in 3D
    :param cfg:             DataConfig object defining configuration
    :param ax:              The Matplotlib axis object
    :param fluid_v0:        Fluid velocity along first sensor array
    :param fluid_v1:        Fluid velocity along second sensor array
    :param xs:              The x-coordinates of all spheres
    :param ys:              The y-coordinates af all spheres
    :param zs:              The z-coordinates of all spheres
    :param x_mesh3d:        3d mesh of x-values
    :param y_mesh3d:        3d mesh of y-values
    :param z_mesh3d:        3d mesh of z-values
    :param latline:         Lateral line instance
    :param s:               The domain for which to plot the excitation
    :param target0:         The first target matrix
    :param target1:         The second target matrix
    """
    x_slice = x_mesh3d[:, :, 0]
    column_indices, row_indices0, row_indices0_mod, row_indices1, row_indices1_mod = get_index_arrays(
        cfg, x_slice, y_mesh3d, z_mesh3d
    )

    # Initialize the plot
    init_plot(ax, cfg.sensor_range, cfg.z_range)

    # Compute the width of the bars
    width = (cfg.sensor_range[1] - cfg.sensor_range[0]) / cfg.n_sensors

    # Plot the excitation pattern
    plot_excitation(ax, fluid_v0, fluid_v1, s, width)

    # Plot the target spheres
    plot_targets(ax, xs, ys, zs)

    # Plot the reconstruction of the spheres using the target matrices
    plot_reconstruction(ax, target0, target1, x_mesh3d, column_indices, row_indices0_mod, row_indices0,
                        row_indices1_mod, row_indices1, y_mesh3d, z_mesh3d, cfg.resolution)

    plot_velocity_arrows(ax, latline, xs, ys, zs)

    # Use some plotting proxies to be able to display the legend
    scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
    ax.legend([scatter1_proxy, scatter2_proxy], ['Reconstructed', 'Target'], numpoints=1)

    plt.pause(0.02)


def get_index_arrays(cfg, x_slice, y_mesh3d, z_mesh3d):
    # Compute the conversion of the mesh grid to the indices in the target matrices
    column_indices = distance_to_idx(x_slice, cfg.sensor_range, cfg.n_sensors).astype(np.int64)
    # Compute the conversion of the mesh grid to the z-indices in the ta
    row_indices0 = distance_to_idx(np.sqrt((y_mesh3d + .5) ** 2 + z_mesh3d ** 2), cfg.z_range, cfg.resolution)
    row_indices1 = distance_to_idx(np.sqrt((y_mesh3d - .5) ** 2 + z_mesh3d ** 2), cfg.z_range, cfg.resolution)
    row_indices0_mod = np.mod(row_indices0, 1)
    row_indices1_mod = np.mod(row_indices1, 1)
    row_indices0 = row_indices0.astype(np.int64)
    row_indices1 = row_indices1.astype(np.int64)
    return column_indices, row_indices0, row_indices0_mod, row_indices1, row_indices1_mod


def plot_velocity_arrows(ax, latline, xs, ys, zs):
    """
    Plots the arrows indicating the velocity components of the spheres
    :param ax:      The matplotlib axis object
    :param latline: The lateral line object
    :param xs:      The spheres' x-coordinates
    :param ys:      The spheres' y-coordinates
    :param zs:      The spheres' z-coordinates
    """
    vx, vy, vz = latline.spheres[0].get_velocity()
    ax.quiver(xs[0], ys[0], zs[0], vx, vy, vz, length=.2, pivot='tail')


def init_plot(ax, sensor_range, z_range):
    """
    Initializes the plot
    :param ax:              The matplotlib axis object
    :param sensor_range:    The range at which the sensors are placed
    :param z_range:         The range of the z-values
    """
    ax.cla()
    ax.set_xlim(sensor_range[0], sensor_range[1])
    ax.set_ylim(sensor_range[0], sensor_range[1])
    ax.set_zlim(z_range[0], z_range[1])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')


def plot_reconstruction(ax, target0, target1, x_mesh3d, col_indices, row_indices0_mod, row_indices0, row_indices1_mod,
                        row_indices1, y_mesh3d, z_mesh3d, resolution):
    """
    Plots the reconstruction of the spheres from the target matrices.
    :param ax:                  The matplotlib axis object
    :param target0:             The first target matrix
    :param target1:             The second target matrix
    :param x_mesh3d:            3d mesh of x-coordinates
    :param col_indices:         The column indices of the target matrices
    :param row_indices0_mod:    The mod of the row indices per z-slice for the first target matrix
    :param row_indices0:        The row indices per z-slice for the first target matrix
    :param row_indices1_mod:    The mod of the row indices per z-slice for the second target matrix
    :param row_indices1:        The row indices per z-slice for the second target matrix
    :param y_mesh3d:            3d mesh of y-coordinates
    :param z_mesh3d:            3d mesh of z-coordinates
    :param resolution:          The resolution of the target matrices
    :return:
    """
    multis = get_3d_density(col_indices, resolution, row_indices0, row_indices0_mod, row_indices1, row_indices1_mod,
                            target0, target1)  # Now we perform a connected components operation
    multis = np.array(multis).transpose((1, 2, 0))
    label_im, nb_labels = ndimage.label(multis > .8)
    if nb_labels >= 1:
        # If there are any connected components surviving the threshold of 0.8  we will reconstruct a sphere form that:
        xs_hat = np.array(ndimage.mean(x_mesh3d, label_im, range(1, nb_labels + 1)))
        ys_hat = np.array(ndimage.mean(y_mesh3d, label_im, range(1, nb_labels + 1)))
        zs_hat = np.array(ndimage.mean(z_mesh3d, label_im, range(1, nb_labels + 1)))
        ax.scatter(xs_hat, ys_hat, zs_hat, c='b', s=60)
        draw_helper_lines(ax, xs_hat, ys_hat, zs_hat, style='b:')


def get_3d_density(col_indices, resolution, row_indices0, row_indices0_mod, row_indices1, row_indices1_mod, target0,
                   target1):
    multis = []
    for i in range(resolution):
        # Get the row indices of the current slice
        ri0 = row_indices0[:, :, i]
        ri1 = row_indices1[:, :, i]

        # Also use the closest other index for interpolation
        rmod0 = row_indices0_mod[:, :, i]
        rmod1 = row_indices1_mod[:, :, i]

        # Take the right entries from the target matrices and interpolate for better accuracy
        m = (
        (target0[ri0, col_indices] * (1 - rmod0) + target0[np.minimum(ri0 + 1, resolution - 1), col_indices] * rmod0) *
        (target1[ri1, col_indices] * (1 - rmod1) + target1[np.minimum(ri1 + 1, resolution - 1), col_indices] * rmod1))

        # Append element-wise multiplication of the target matrices
        multis.append(m)

    return multis


def draw_helper_lines(ax, xs, ys, zs, style='r:'):
    """
    Draws helper lines in the plot to better visualize where the spheres are w.r.t. to the axes of the plot
    :param ax:      The Matplotlib axis object
    :param xs:      The x-coordinates of the spheres
    :param ys:      The y-coordinates of the spheres
    :param zs:      The z-coordinates of the spheres
    :param style:   The drawing style
    """
    for x, y, z in zip(xs, ys, zs):
        ax.plot([x, x], [0, y], style)
        ax.plot([0, x], [y, y], style)
        ax.plot([x, x], [y, y], [0, z], style)


def plot_targets(ax, xs, ys, zs):
    """
    This function plots the target spheres.
    :param ax:  The matplotlib axis object
    :param xs:  The x-coordinates of the spheres
    :param ys:  The y-coordinates of the spheres
    :param zs:  The z-coordinates of the spheres
    """
    ax.scatter(xs, ys, zs, c='r', s=60)
    draw_helper_lines(ax, xs, ys, zs, 'r:')


def plot_excitation(ax, fluid_v0, fluid_v1, s, width):
    """
    Plot the excitation patterns measured at the sensor arrays.
    :param ax:          The matplotlib axis object
    :param fluid_v0:    Fluid velocity along first array
    :param fluid_v1:    Fluid velocity along second array
    :param s:           The domain to use for excitation
    :param width:       The width of the bars
    """
    ax.bar(s, fluid_v0 * 5000, zs=-.5, zdir='y', color='b', width=width, alpha=0.5)
    ax.bar(s, fluid_v1 * 5000, zs=.5, zdir='y', color='r', width=width, alpha=0.5)


def distance_to_idx(d, range_d, maxi):
    idx = np.asarray((d - range_d[0]) / (range_d[1] - range_d[0]) * maxi)
    return np.minimum(np.maximum(idx, 0), maxi-1)