import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
from scipy import ndimage
import os
from mpl_toolkits.mplot3d import Axes3D

from latline.experiment_config import DataConfig, parse_config_args
from latline.latline import Latline

from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':14})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def distanceToIdx(d, range_d, maxi):
    idx = np.asarray((d - range_d[0]) / (range_d[1] - range_d[0]) * maxi)
    return np.minimum(np.maximum(idx, 0), maxi-1)


def generateData(N, x_range, y_range, z_range, v, d_theta_range, z_res, sigma, N_sensors, sensor_range,
                 display, tau=2):
    # pre-alloc data and labels
    data    = np.zeros((N, 2, N_sensors, tau))
    targets  = np.zeros((N, 2, N_sensors, z_res))
    zz_labels = np.zeros((N, N_sensors, N_sensors, z_res))

    # s should be interpreted as in Boulogne et al (2015)
    s = np.linspace(sensor_range[0], sensor_range[1], N_sensors)
    x_grid = s
    y_grid = s
    z_grid = np.linspace(z_range[0], z_range[1], z_res)

    # xx and yy are matrices for evaluating a Gaussian that is centered at a sphere
    # xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)

    xx, fx = np.meshgrid(x_grid, z_grid)
    yy, fy = np.meshgrid(y_grid, z_grid)

    xxx, yyy, zzz = np.meshgrid(x_grid, y_grid, z_grid)

    yyy1 = yyy - (-.5)
    yyy2 = yyy - .5

    # Init lateral line for train data
    latline = Latline(x_range=x_range, y_range=y_range, z_range=z_range, d_theta_range=d_theta_range,
                      sensor_range=sensor_range, n_sensors=N_sensors, min_spheres=1, max_spheres=2, v=v)

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax2 = fig.add_subplot(312)
        #ax3 = fig.add_subplot(313)

    buff_x = []
    buff_y = []

    # Loop through all data points
    for i in range(N + tau):
        if i % 1000 == 0:
            print('current iteration: ', i)

        # Obtain x and y locations of all spheres and the corresponding sensor measurements
        xs, ys, zs, fluid_v_x, fluid_v_y = latline.step()  # xs' rows correspond to the spheres

        # Compute each Gaussian separately
        fx1_ = np.asarray([np.exp(-((xx - xi) ** 2 + (fx - np.sqrt((yi+0.5) ** 2 + zi ** 2)) ** 2) / (2 * sigma**2))
                          for (xi, yi, zi) in zip(xs, ys, zs)])
        fx2_ = np.asarray([np.exp(-((xx - xi) ** 2 + (fx - np.sqrt((yi-0.5) ** 2 + zi ** 2)) ** 2) / (2 * sigma**2))
                          for (xi, yi, zi) in zip(xs, ys, zs)])
        # And then take the maximum over all spheres.
        fx1_ = np.max(fx1_, axis=0)
        fx2_ = np.max(fx2_, axis=0)

        # Save measurements and heatmap
        buff_x.append(fluid_v_x)
        buff_y.append(fluid_v_y)

        #print fx_.shape
        if i >= tau:
            del buff_x[0]
            del buff_y[0]

            data[i - tau][0] = np.transpose(buff_x)
            data[i - tau][1] = np.transpose(buff_y)
            targets[i - tau][0] = np.transpose(fx1_.reshape((z_res, N_sensors)))
            targets[i - tau][1] = np.transpose(fx2_.reshape((z_res, N_sensors)))

        if display:

            x_slice = xxx[:, :, 0]

            xidxs = distanceToIdx(x_slice, sensor_range, N_sensors).astype(np.int64)

            y1zidxs = distanceToIdx(np.sqrt((yyy + .5) ** 2 + zzz ** 2), z_range, z_res)
            y2zidxs = distanceToIdx(np.sqrt((yyy - .5) ** 2 + zzz ** 2), z_range, z_res)

            y1z_idxs_mod = np.mod(y1zidxs, 1)
            y2z_idxs_mod = np.mod(y2zidxs, 1)

            y1zidxs = y1zidxs.astype(np.int64)
            y2zidxs = y2zidxs.astype(np.int64)

            if plot_3d:
                ax.cla()
                ax.scatter(xs, ys, zs, c='r', s=60)

                ax.set_xlim(sensor_range[0], sensor_range[1])
                ax.set_ylim(sensor_range[0], sensor_range[1])
                ax.set_zlim(z_range[0], z_range[1])

                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$y$')
                ax.set_zlabel(r'$z$')

                width = (sensor_range[1] - sensor_range[0]) / N_sensors

                ax.bar(s, fluid_v_x * 5, zs=-.5, zdir='y', color='b', width=width, alpha=0.5)
                ax.bar(s, fluid_v_y * 5, zs=.5, zdir='y', color='r', width=width, alpha=0.5)

                #ax2.pcolormesh(xx, fx, fx_)
                #ax3.pcolormesh(yy, fy, fy_)

            multis = []
            for i in range(z_res):
                z1i = y1zidxs[:, :, i]
                z2i = y2zidxs[:, :, i]

                z1mod = y1z_idxs_mod[:, :, i]
                z2mod = y2z_idxs_mod[:, :, i]

                mult = ((fx1_[z1i, xidxs] * (1-z1mod) + fx1_[np.minimum(z1i + 1, z_res - 1), xidxs] * z1mod) / 1 *
                        (fx2_[z2i, xidxs] * (1-z2mod) + fx2_[np.minimum(z2i + 1, z_res - 1), xidxs] * z2mod) / 1)

                multis.append(mult)

            multis = np.array(multis).transpose((1, 2, 0))
            label_im, nb_labels = ndimage.label(multis > .8)

            #print label_im
            #print xxx.shape
            #print 'detected objects from reconstruction: ', nb_labels

            if nb_labels >= 1:
                sizes = np.array(ndimage.sum(multis > .8, label_im, range(nb_labels + 1)))
                #print sizes
                '''
                xs_hat = np.array(ndimage.sum(multis * xxx, label_im, range(1, nb_labels + 1))) / sizes
                ys_hat = np.array(ndimage.sum(multis * yyy, label_im, range(1, nb_labels + 1))) / sizes
                zs_hat = np.array(ndimage.sum(multis * zzz, label_im, range(1, nb_labels + 1))) / sizes
                '''
                xs_hat = np.array(ndimage.mean(xxx, label_im, range(1, nb_labels + 1)))
                ys_hat = np.array(ndimage.mean(yyy, label_im, range(1, nb_labels + 1)))
                zs_hat = np.array(ndimage.mean(zzz, label_im, range(1, nb_labels + 1)))

                #print xs, xs_hat
                #print ys, ys_hat
                #print zs, zs_hat

                ax.scatter(xs_hat, ys_hat, zs_hat, c='b', s=60)
                for x, y, z in zip(xs_hat, ys_hat, zs_hat):
                    ax.plot([x, x], [0, y], 'b:')
                    ax.plot([0, x], [y, y], 'b:')
                    ax.plot([x, x], [y, y], [0, z], 'b:')

                for x, y, z in zip(xs, ys, zs):
                    ax.plot([x, x], [0, y], 'r:')
                    ax.plot([0, x], [y, y], 'r:')
                    ax.plot([x, x], [y, y], [0, z], 'r:')

            scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
            scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')

            vx, vy, vz = latline.spheres[0].get_velocity()
            ax.quiver(xs[0], ys[0], zs[0], vx, vy, vz, length=.2, pivot='tail')
            #ax.legend([scatter1_proxy, scatter2_proxy], [r'Reconstructed', r'Actual'], numpoints=1)


            #plt.savefig('tex_plot.eps')
            #plt.tight_layout()
            #plt.show()
            plt.pause(0.02)

    return data, targets, zz_labels

if __name__ == "__main__":
    config = DataConfig(parse_config_args(mode='data'))

    # Sigma for Gaussians around spheres.
    sigma = 0.2

    # Obtain data for train, test and validation sets
    train_data, train_labels, zz_    = generateData(N=config.N_train, x_range=config.x_range, y_range=config.y_range,
                                                    z_range=config.z_range, v=config.v, d_theta_range=config.d_theta_range,
                                                    z_res=config.z_res, sigma=config.sigma, N_sensors=config.N_sensors,
                                                    sensor_range=config.sensor_range, display=config.display,
                                                    tau=config.tau)
    test_data, test_labels, _      = generateData(N=config.N_test, x_range=config.x_range, y_range=config.y_range,
                                                  z_range=config.z_range, v=config.v, d_theta_range=config.d_theta_range,
                                                  z_res=config.z_res, sigma=config.sigma, N_sensors=config.N_sensors,
                                                  sensor_range=config.sensor_range, display=config.display,
                                                  tau=config.tau)
    val_data, val_labels, _        = generateData(N=config.N_val, x_range=config.x_range, y_range=config.y_range,
                                                  z_range=config.z_range, v=config.v, d_theta_range=config.d_theta_range,
                                                  z_res=config.z_res, sigma=config.sigma, N_sensors=config.N_sensors,
                                                  sensor_range=config.sensor_range, display=config.display,
                                                  tau=config.tau)

    print('Max: ', np.max(train_data), np.min(train_data))
    print('Mean: ', np.mean(train_data))

    project_folder = os.path.dirname(os.path.realpath(__file__))

    # Serialize the data
    os.makedirs(os.path.join(project_folder, 'data'), exist_ok=True)
    with open(os.path.join(project_folder, 'data', 'multisphere_parallel.pickle'), 'wb') as f:
        pickle.dump([train_data, train_labels, test_data, test_labels, val_data, val_labels], f)
