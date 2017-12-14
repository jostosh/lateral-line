from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import numpy as np
import tensorflow as tf

from latline.common.data_util import DataBatcher, read_data
from latline.experiment_config import ExperimentConfig, init_log_dir
from latline.models import define_train_step, define_loss, create_model
import argparse
import ast
import matplotlib.pylab as plt
from latline.experiment_config import DataConfig
from generate_data import get_meshes
from latline.common.plot3d import get_index_arrays, get_3d_density
import time
import visvis as vv  # doctest: +SKIP

f = vv.figure(1)
vv.clf()

a1 = vv.subplot(121)
a2 = vv.subplot(122)
a1.camera = a2.camera


def demo(restore_path):
    """
    Trains a convolutional neural network to locate multiple spheres in a simulated artificial lateral line experiment.
    Currently, this implementation is a port from the original Theano implementation, which is why it still misses some
    functionality that is mentioned in the current version of the paper.
    :param config:  Experiment config generated from command line input.
    """

    # Create TensorFlow session
    sess = tf.Session()

    # Read the data from storage
    test_x, test_y, _, _ = read_data(config)
    print("Finished reading data")

    # Set up two data batchers that provide a straightforward interface for moving through data batches
    excitation0, excitation1, out = create_model(config, test_x, test_y, mode='test')

    # Initialize the graph variables
    sess.run(tf.global_variables_initializer())

    # Create a saver
    saver = tf.train.Saver()

    saver.restore(sess, restore_path)

    out_numeric = sess.run(out, feed_dict={
       excitation0: test_x[0:1, 0, :, :],
       excitation1: test_x[0:1, 1, :, :]
    })
    # fig, ax = plt.subplots(2, 2)
    #
    # plt.ion()
    # plt.show()

    DataConfig.load()
    ex_cfg = DataConfig
    ex_cfg.resolution = 128
    ex_cfg.n_sensors = 128
    s, target_mesh, x_mesh2d, x_mesh3d, y_mesh3d, z_mesh3d = get_meshes()

    x_slice = x_mesh3d[:, :, 0]
    column_indices, row_indices0, row_indices0_mod, row_indices1, row_indices1_mod = get_index_arrays(
        ex_cfg, x_slice, y_mesh3d, z_mesh3d
    )
    ax_objs = None

    counter = 0
    n_angles = 360
    print("Running animation")
    for i in range(test_x.shape[0]):
        out_numeric = sess.run(out, feed_dict={
            excitation0: test_x[i:i+1, 0, :, :],
            excitation1: test_x[i:i+1, 1, :, :]
        })
        out_numeric = np.transpose(out_numeric[0])

        halfway = out_numeric.shape[0] // 2

        multis = get_3d_density(column_indices, ex_cfg.resolution, row_indices0, row_indices0_mod, row_indices1,
                                row_indices1_mod, out_numeric[:halfway, :], out_numeric[halfway:, :])
        target = get_3d_density(column_indices, ex_cfg.resolution, row_indices0, row_indices0_mod, row_indices1,
                                row_indices1_mod, test_y[i, 0].T, test_y[i, 1].T)
        target3d = np.asarray(target)
        density3d = np.asarray(multis)

        # def display():
        #     contour3d(x_mesh3d, y_mesh3d, z_mesh3d, density3d, contours=[0.8], transparent=True)
        # display()

        a1.Clear()
        a2.Clear()
        vv.volshow(density3d, cm=vv.CM_JET, axes=a1)
        vv.volshow(target3d, cm=vv.CM_JET, axes=a2)

        if density3d.max() > args.level:
            mesh = vv.isosurface(density3d, args.level)
            m = vv.mesh(mesh, axes=a1)
            m.faceColor = (1, 1, 1)

        if target3d.max() > args.level:
            mesh = vv.isosurface(target3d, args.level)
            m = vv.mesh(mesh, axes=a2)
            m.faceColor = (1, 1, 1)


        a1.daspect = (3, 3, 2)
        a1.axis.xLabel = 'x'
        a1.axis.yLabel = 'y'
        a1.axis.zLabel = 'z'
        vv.title('Prediction reconstruction', axes=a1)

        a2.daspect = (3, 3, 2)
        a2.axis.xLabel = 'x'
        a2.axis.yLabel = 'y'
        a2.axis.zLabel = 'z'
        vv.title('Target reconstruction', axes=a2)

        for i in range(5):
            a1.camera.azimuth = counter % 360
            counter += 3

            a1.Draw()
            a2.Draw()
            f.DrawNow()
            if i < 4:
                time.sleep(0.2)
        # exit(0)

        #
        # if not ax_objs:
        #     ax_objs = []
        #     ax_objs.append([
        #         ax[0, 0].imshow(out_numeric[:halfway, :], cmap='viridis'),
        #         ax[0, 1].imshow(out_numeric[halfway:, :], cmap='viridis')
        #     ])
        #     ax_objs.append([
        #         ax[1, 0].imshow(np.transpose(test_y[i, 0]), cmap='viridis'),
        #         ax[1, 1].imshow(np.transpose(test_y[i, 1]), cmap='viridis')
        #     ])
        # else:
        #     ax_objs[0][0].set_data(out_numeric[:halfway, :])
        #     ax_objs[0][1].set_data(out_numeric[halfway:, :])
        #     ax_objs[1][0].set_data(np.transpose(test_y[i, 0]))
        #     ax_objs[1][1].set_data(np.transpose(test_y[i, 1]))
        #
        # plt.tight_layout()
        # plt.pause(1/15)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_path", type=str, default='tensorboardlogs/v0.5/test/run0002/model.ckpt')
    parser.add_argument("--data", default='data')
    parser.add_argument("--fnm", default='the_fourth')
    parser.add_argument("--level", default=0.75, type=float)
    args = parser.parse_args()

    def read_config():
        with open(os.path.join(os.path.dirname(args.restore_path), 'config.txt'), 'r') as f:
            s = f.read()
        return ast.literal_eval(s)


    config = ExperimentConfig()
    for p, v  in read_config().items():
        config.__setattr__(p, v)
    demo(restore_path=args.restore_path)
