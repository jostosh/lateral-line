import os
import pickle

import numpy as np
import tensorflow as tf

from common.data_util import DataBatcher
from latline.experiment_config import ExperimentConfig, parse_config_args, init_log_dir
from latline.layers import define_multi_range_input, noise_layer
from latline.models import define_train_step, define_loss, define_cross_network, define_parallel_network, define_inputs


def train(config):
    """
    Trains a convolutional neural network to locate multiple spheres in a simulated artificial lateral line experiment.
    Currently, this implementation is a port from the original Theano implementation, which is why it still misses some
    functionality that is mentioned in the current version of the paper.
    :param config:  Experiment config generated from command line input.
    """

    sensor_index = 1
    slice_index = 3

    # Create TensorFlow session
    sess = tf.Session()

    # Read the data from storage
    test_x, test_y, train_x, train_y = read_data(config)

    # Set up two data batchers that provide a straightforward interface for moving through data batches
    data_batcher_train = DataBatcher(train_x, train_y, config.batch_size)
    data_batcher_test = DataBatcher(test_x, test_y, len(test_x))

    # We take the output depth from the target data
    output_depth = train_y.shape[slice_index] * train_y.shape[sensor_index]
    # This should be added to our list of n_kernels
    config.n_kernels.append(output_depth)
    # Also add this to list containing the number of neurons in fully connected layers
    config.n_units.append(np.prod(train_y.shape[sensor_index:]))

    # Now we can set up the network. First we define the input placeholders
    excitation0, excitation1 = define_inputs((None,) + train_x.shape[-2:])

    data_std_dev = np.std(np.concatenate((test_x, train_x)))

    # Optionally use inputs with different ranges
    excitation0_preprocessed, excitation1_preprocessed = \
        noise_layer(data_std_dev * config.noise, excitation0), noise_layer(data_std_dev * config.noise, excitation1)
    if config.multi_range:
        excitation0_preprocessed, excitation1_preprocessed = define_multi_range_input(
            config.multi_range_trainable, config.input_factors, excitation0, excitation1
        )

    # Then, we define the network giving us the output
    out = {
        'parallel': define_parallel_network,
        'cross': define_cross_network
    }[config.model](config, excitation0_preprocessed, excitation1_preprocessed)

    # Next, we define, the loss that depends on the error + some L2 regularization of the weights
    # For reporting performance, the MSE is used
    loss, mse, target = define_loss(out, train_y, config.loss)

    # We define the train step which is the Op that should be called when training.
    train_step, global_step = define_train_step(loss, config.lr, config.optimizer)

    # Initialize writer for TensorBoard logging
    logdir = init_log_dir(config)
    print("Saving results to {}".format(logdir))
    train_writer    = tf.summary.FileWriter(os.path.join(logdir, 'train'), sess.graph)
    test_writer     = tf.summary.FileWriter(os.path.join(logdir, 'test'), sess.graph)
    train_summary   = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES + '/train'))
    test_summary    = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES + '/test'))

    # Set the number of batches inside a single train iteration
    n_batches = int(np.ceil(train_x.shape[0] / config.batch_size))

    # Initialize the graph variables
    sess.run(tf.global_variables_initializer())

    with open(os.path.join(logdir, 'results.csv'), 'w') as f:
        f.write('epoch,mse,loss\n')
        # Now we start training!
        for epoch in range(config.n_epochs):
            for batch_index in range(n_batches):
                batch_x, batch_y = data_batcher_train.next_batch()
                fdict = {excitation0: batch_x[:, 0, :, :], excitation1: batch_x[:, 1, :, :], target: batch_y}
                if batch_index % 10 == 0:
                    mean_square_error, l, _, summ, step = sess.run(
                        [mse, loss, train_step, train_summary, global_step],
                        feed_dict=fdict)
                    train_writer.add_summary(summ, global_step=step)
                else:
                    mean_square_error, l, _ = sess.run(
                        [mse, loss, train_step],
                        feed_dict=fdict)

            test_x, test_y = data_batcher_test.next_batch()

            mean_square_error, summ, step, loss_num = sess.run(
                [mse, test_summary, global_step, loss],
                feed_dict={excitation0: test_x[:, 0, :, :], excitation1: test_x[:, 1, :, :], target: test_y}
            )
            test_writer.add_summary(summ, global_step=step)
            print("Epoch ({}/{}), Test MSE: {}".format(epoch, config.n_epochs, mean_square_error))
            f.write(str(epoch) + ',' + str(mean_square_error) + ',' + str(loss_num) + '\n')
            f.flush()

def read_data(config):
    """
    Reads in the data
    """
    path = os.path.join(config.data, "{}.pickle".format(config.fnm))
    print("Reading data at {}".format(path))
    with open(path, 'rb') as f:
        train_x, train_y, test_x, test_y = pickle.load(f)
    return test_x, test_y, train_x, train_y


if __name__ == "__main__":
    config = ExperimentConfig(parse_config_args())
    train(config)
