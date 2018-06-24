import os

import numpy as np
import tensorflow as tf

from generate_data import get_meshes
from latline.common.data_util import DataBatcher, read_data
from latline.common.plot3d import get_index_arrays, get_3d_density
from latline.common.tf_utils import init_summary_writer
from latline.experiment_config import ExperimentConfig, init_log_dir, DataConfig
from latline.models import define_train_step, define_loss, create_model
from latline.util.config import setup_logdir
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import distance_matrix
import json
from tqdm import trange
import scipy.stats as stats


def train(config):
    """
    Trains a convolutional neural network to locate multiple spheres in a simulated artificial
    lateral line experiment. Currently, this implementation is a port from the original Theano
    implementation, which is why it still misses some functionality that is mentioned in the
    current version of the paper.
    :param config:  Experiment config generated from command line input.
    """
    # Read the data from storage
    # test_x, test_y, train_x, train_y = read_data(config)
    print("Reading data")
    train_excitations, train_targets, train_xyz, test_excitations, test_targets, test_xyz = \
        read_data(config)
    # test_x = test_x[:, :, :, -config.tau:]
    # train_x = train_x[:, :, :, -config.tau:]
    train_excitations = train_excitations[:, :, :, -config.tau:]
    test_excitations = test_excitations[:, :, :, -config.tau:]

    # Set up two data batchers that provide a straightforward interface for moving
    # through data batches
    data_batcher_train = DataBatcher(config.batch_size, train_excitations, train_targets, train_xyz)
    data_batcher_test = DataBatcher(config.batch_size, test_excitations, test_targets, test_xyz)

    print("Creating model")
    exc0_ph, exc1_ph, logits, density_pred_op = create_model(
        config, train_excitations, train_targets)

    # Next, we define the loss that depends on the error + some L2 regularization of the weights
    # For reporting performance, the MSE is used
    loss_op, mse_op, target_ph = define_loss(logits, train_targets, config.loss)

    # We define the train step which is the Op that should be called when training.
    train_op, global_step = define_train_step(loss_op, config.lr, config.optimizer)

    # Set up the localizer
    localizer = Localizer()

    # Initialize writer for TensorBoard logging
    logdir = setup_logdir(
        ExperimentConfig.log_base, ExperimentConfig.version, ExperimentConfig.log_sub)
    print("Saving results to {}".format(logdir))
    with open(os.path.join(logdir, "params.json"), 'w') as f:
        json.dump({k: str(v) for k, v in ExperimentConfig.__dict__.items()}, fp=f, indent=4)

    with tf.Session() as sess:
        train_summary, train_writer = init_summary_writer(logdir, sess, 'train')
        test_summary, test_writer = init_summary_writer(logdir, sess, 'test')

        # Set the number of batches inside a single train iteration

        # Initialize the graph variables
        sess.run(tf.global_variables_initializer())

        # Create a saver
        print("Creating saver")
        saver = tf.train.Saver()

        logpath = os.path.join(logdir, 'results.csv')
        print("Logging at {}".format(logpath))
        with open(logpath, 'w') as f:
            f.write('epoch,mse,loss,avg_dist,avg_dist_transpose,mean_low,mean_high,count_correct\n')
            # Now we start training!
            pbar_epoch = trange(config.n_epochs)
            test_losses = None
            for epoch in pbar_epoch:
                pbar_epoch.set_description("Epoch={0} | Loss={1:.4f}".format(
                    str(epoch).zfill(int(np.log10(config.n_epochs)) + 1),
                    np.mean(test_losses) if test_losses is not None else np.nan))
                pbar_batch = data_batcher_train.iterate_one_epoch()
                for batch_index in pbar_batch:
                    batch_excitations, batch_targets, batch_xyz = \
                        data_batcher_train.next_batch()
                    fdict = {
                        exc0_ph: batch_excitations[:, 0, :, :],
                        exc1_ph: batch_excitations[:, 1, :, :],
                        target_ph: batch_targets
                    }
                    if batch_index % 50 == 0:
                        mse_out, loss_out, _, summ, step = sess.run(
                            [mse_op, loss_op, train_op, train_summary, global_step],
                            feed_dict=fdict
                        )
                        train_writer.add_summary(summ, global_step=step)
                    else:
                        mse_out, loss_out, _ = sess.run(
                            [mse_op, loss_op, train_op], feed_dict=fdict)
                    pbar_batch.set_description("Loss={0:.4f}".format(loss_out))

                # Test the whole
                test_losses, test_mses = [], []
                test_count_low, test_count_high, test_count_correct, test_dists, \
                    test_dists_transpose = [], [], [], [], []
                pbar_batch_test = data_batcher_test.iterate_one_epoch()
                for _ in pbar_batch_test:
                    test_excitations, test_targets, test_xyz = data_batcher_test.next_batch()

                    mse_out, step, loss_out, density_pred_out = sess.run(
                        fetches=[mse_op, global_step, loss_op, density_pred_op],
                        feed_dict={
                            exc0_ph: test_excitations[:, 0, :, :],
                            exc1_ph: test_excitations[:, 1, :, :],
                            target_ph: test_targets
                        }
                    )
                    test_losses.append(loss_out)
                    test_mses.append(mse_out)

                    if epoch >= config.n_epochs - 1:
                        count_low, count_high, count_correct, mean_dist, mean_dist_transpose = \
                            localizer.localize(density_pred_out, test_xyz)
                        test_count_low.extend(count_low)
                        test_count_high.extend(count_high)
                        test_count_correct.append(count_correct)
                        test_dists.append(mean_dist)
                        test_dists_transpose.append(mean_dist_transpose)
                test_mse = np.mean(test_mses)
                test_loss = np.mean(test_losses)
                test_cnt_low = np.mean(test_count_low)
                test_cnt_high = np.mean(test_count_high)
                test_cnt_correct = np.mean(test_count_correct)
                test_dist = np.mean(test_dists)
                test_dist_transpose = np.mean(test_dists_transpose)
                f.write(','.join([str(n) for n in [
                    epoch, test_mse, test_loss, test_dist, test_dist_transpose,
                    test_cnt_low, test_cnt_high, test_cnt_correct]]) + '\n')
                f.flush()

                if (epoch + 1) % 50 == 0:
                    save_path = saver.save(sess, os.path.join(logdir, 'model.ckpt'))
                    print("Stored weights at:", save_path)


class Localizer:

    def __init__(self):
        s, target_mesh, x_mesh2d, self.x_mesh3d, self.y_mesh3d, self.z_mesh3d = get_meshes()
        x_slice = self.x_mesh3d[:, :, 0]
        self.col_ind, \
        self.row_ind0, self.row_ind0_mod, \
        self.row_ind1, self.row_ind1_mod = \
            get_index_arrays(DataConfig, x_slice, self.y_mesh3d, self.z_mesh3d)

        self.x_mesh2d, self.z_mesh2d = np.meshgrid(
            np.linspace(*DataConfig.x_range, num=DataConfig.n_sensors),
            np.linspace(*DataConfig.z_range, num=DataConfig.resolution))

        self.z_mesh_array0 = np.sqrt((self.z_mesh3d ** 2 + (self.y_mesh3d - 0.5) ** 2))
        self.z_mesh_array1 = np.sqrt((self.z_mesh3d ** 2 + (self.y_mesh3d + 0.5) ** 2))

        self.xz_mesh_array0 = np.stack((self.x_mesh3d, self.z_mesh_array0), axis=-1)
        self.xz_mesh_array1 = np.stack((self.x_mesh3d, self.z_mesh_array1), axis=-1)

    @staticmethod
    def gaussian_connected_components(density):
        filtered = gaussian_filter(density, DataConfig.sigma, mode='constant')
        connected_components, num_labels = ndimage.label(
            np.greater(filtered, config.localization_threshold))
        return connected_components, num_labels

    def localize(self, density, xyz):
        density0, density1 = density[..., :config.resolution], density[..., config.resolution:]

        localization_count_diffs = []
        localization_errors = []
        localization_errors_transpose = []
        for d0, d1, (xs, ys, zs) in zip(density0, density1, xyz):

            density_3d = get_3d_density(
                self.col_ind, config.resolution,
                self.row_ind0, self.row_ind0_mod,
                self.row_ind1, self.row_ind1_mod,
                d0.transpose(), d1.transpose())
            filtered = gaussian_filter(density_3d, DataConfig.sigma, mode='constant')
            thresholded = np.greater(filtered, config.localization_threshold)
            connected_components, num_labels = ndimage.label(thresholded.transpose((1, 2, 0)))

            if num_labels >= 1:
                xs_hat = np.array(ndimage.mean(
                    self.x_mesh3d, connected_components, range(1, num_labels + 1)))
                ys_hat = np.array(ndimage.mean(
                    self.y_mesh3d, connected_components, range(1, num_labels + 1)))
                zs_hat = np.array(ndimage.mean(
                    self.z_mesh3d, connected_components, range(1, num_labels + 1)))

                prediction_mat = np.stack(zip(xs_hat, ys_hat, zs_hat))
                target_mat = np.stack(zip(xs, ys, zs))

                distance_mat = distance_matrix(target_mat, prediction_mat)

                distance_ind = np.argsort(distance_mat, axis=1)
                if len(np.unique(distance_ind[:, 0])) == len(xs):
                    distance_average = np.mean(np.min(distance_mat, axis=1))
                else:
                    distance_sort = np.sort(distance_mat, axis=1)
                    winning_first = np.argmin(distance_sort[:, 0])
                    distance_average = (distance_sort[winning_first, 0] +
                                        distance_sort[1 - winning_first, 1 if num_labels > 1 else 0]
                                        ) / 2
                localization_count_diffs.append(num_labels - len(xs))
                localization_errors.append(distance_average)

                # Turn around ys and xs....
                target_mat = np.stack(zip(ys, xs, zs))

                distance_mat = distance_matrix(target_mat, prediction_mat)

                distance_ind = np.argsort(distance_mat, axis=1)
                if len(np.unique(distance_ind[:, 0])) == len(xs):
                    distance_average = np.mean(np.min(distance_mat, axis=1))
                else:
                    distance_sort = np.sort(distance_mat, axis=1)
                    winning_first = np.argmin(distance_sort[:, 0])
                    distance_average = (distance_sort[winning_first, 0] +
                                        distance_sort[1 - winning_first, 1 if num_labels > 1 else 0]
                                        ) / 2
                localization_errors_transpose.append(distance_average)

        localization_errors = np.mean(np.array(localization_errors)) if localization_errors else 0.0
        localization_count_diffs = np.array(localization_count_diffs)

        localization_errors_transpose = np.mean(np.array(localization_errors_transpose)) if localization_errors_transpose else 0.0

        average_larger = localization_count_diffs[localization_count_diffs > 0]
        average_lower = localization_count_diffs[localization_count_diffs < 0]
        average_correct = np.mean(localization_count_diffs == 0)

        return average_lower, average_larger, average_correct, localization_errors, localization_errors_transpose


if __name__ == "__main__":
    ExperimentConfig.load()
    DataConfig.load_defaults()
    config = ExperimentConfig
    train(config)
