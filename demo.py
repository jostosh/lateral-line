import os

import numpy as np
import tensorflow as tf

from latline.common.data_util import DataBatcher, read_data
from latline.experiment_config import ExperimentConfig, parse_config_args, init_log_dir
from latline.models import define_train_step, define_loss, create_model
import argparse
import ast
import matplotlib.pylab as plt
from latline.experiment_config import DataConfig
from generate_data import get_meshes
from latline.common.plot3d import get_index_arrays, get_3d_density
from plotly.tools import FigureFactory as FF

import plotly.plotly as py
import plotly.graph_objs as go

from plotly.grid_objs import Grid, Column
import visvis as vv  # doctest: +SKIP

from skimage import measure


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
    fig, ax = plt.subplots(2, 2)

    plt.ion()
    plt.show()

    ex_cfg = DataConfig()
    s, target_mesh, x_mesh2d, x_mesh3d, y_mesh3d, z_mesh3d = get_meshes(ex_cfg)

    x_slice = x_mesh3d[:, :, 0]
    column_indices, row_indices0, row_indices0_mod, row_indices1, row_indices1_mod = get_index_arrays(
        ex_cfg, x_slice, y_mesh3d, z_mesh3d
    )
    ax_objs = None
    for i in range(test_x.shape[0]):
        out_numeric = sess.run(out, feed_dict={
            excitation0: test_x[i:i+1, 0, :, :],
            excitation1: test_x[i:i+1, 1, :, :]
        })
        out_numeric = np.transpose(out_numeric[0])

        multis = get_3d_density(column_indices, ex_cfg.resolution, row_indices0, row_indices0_mod, row_indices1,
                                row_indices1_mod, out_numeric[:24, :], out_numeric[24:, :])
        density3d = np.transpose(np.asarray(multis), (2, 0, 1))

        verts, faces, normals, values = measure.marching_cubes(density3d, 0.8)
        vv.figure(1)
        vv.clf()
        mesh = vv.mesh(np.fliplr(verts), faces, normals, values)
        mesh.faceColor = (0, 1, 1)
        app = vv.use()
        app.Run()
        exit(0)

        if not ax_objs:
            ax_objs = []
            ax_objs.append([
                ax[0, 0].imshow(out_numeric[:24, :], cmap='viridis'),
                ax[0, 1].imshow(out_numeric[24:, :], cmap='viridis')
            ])
            ax_objs.append([
                ax[1, 0].imshow(np.transpose(test_y[i, 0]), cmap='viridis'),
                ax[1, 1].imshow(np.transpose(test_y[i, 1]), cmap='viridis')
            ])
        else:
            ax_objs[0][0].set_data(out_numeric[:24, :])
            ax_objs[0][1].set_data(out_numeric[24:, :])
            ax_objs[1][0].set_data(np.transpose(test_y[i, 0]))
            ax_objs[1][1].set_data(np.transpose(test_y[i, 1]))

        plt.tight_layout()
        plt.pause(1/15)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_path", type=str, default='tensorboardlogs/test/run0014/model.ckpt')
    parser.add_argument("--data", default='data')
    parser.add_argument("--fnm", default='the_fourth')
    args = parser.parse_args()

    def read_config():
        s = open(os.path.join(os.path.dirname(args.restore_path), 'config.txt'), 'r').read()
        return ast.literal_eval(s)
    config = ExperimentConfig(read_config())

    demo(restore_path=args.restore_path)
