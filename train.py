import os
import pickle

import numpy as np
import tensorflow as tf
from tqdm import trange

from common.data_util import DataBatcher
from latline.experiment_config import ExperimentConfig, parse_config_args, init_log_dir
from latline.models import define_train_step, define_loss, define_cross_network, define_parallel_network, define_inputs


def noise_layer(incoming, std):
    """
    Adds noise to input
    :param incoming: Input tensor
    :param std:      Standard deviation
    :return:         Tensor with added Gaussian noise
    """
    noise = tf.random_normal(shape=tf.shape(incoming), mean=0.0, stddev=std, dtype=tf.float32)
    return incoming + noise


def train(config):
    """
    Trains a convolutional neural network to locate multiple spheres in a simulated artificial lateral line experiment.
    Currently, this implementation is a port from the original Theano implementation, which is why it still misses some
    functionality that is mentioned in the current version of the paper.
    :param config:  Experiment config generated from command line input.
    """

    sensor_index = 1
    spatial_index = 2
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
    config.n_units.append(np.prod(train_y.shape[1:]))

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
    train_step, global_step = define_train_step(loss, config.lr)

    # Initialize writer for TensorBoard logging
    logdir = init_log_dir(config)
    train_writer    = tf.summary.FileWriter(os.path.join(logdir, 'train'), sess.graph)
    test_writer     = tf.summary.FileWriter(os.path.join(logdir, 'test'), sess.graph)
    train_summary   = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES + '/train'))
    test_summary    = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES + '/test'))

    # Set the number of batches inside a single train iteration
    n_batches = int(np.ceil(train_x.shape[0] / config.batch_size))

    # Initialize the graph variables
    sess.run(tf.global_variables_initializer())

    # Now we start training!
    for epoch in range(config.n_epochs):
        t = trange(n_batches)
        t.set_description("Training")
        for batch_index in t:
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
            t.set_postfix(MSE=mean_square_error, loss=l, epoch='{}/{}'.format(epoch, config.n_epochs))

        test_x, test_y = data_batcher_test.next_batch()

        mean_square_error, summ, step = sess.run(
            [mse, test_summary, global_step],
            feed_dict={excitation0: test_x[:, 0, :, :], excitation1: test_x[:, 1, :, :], target: test_y}
        )
        test_writer.add_summary(summ, global_step=step)
        print("Test MSE: {}".format(mean_square_error))


def define_multi_range_input(trainable, input_factors, excitation0, excitation1):
    if trainable:
        input_factors = [tf.Variable(input_fac) for input_fac in input_factors]
        [tf.summary.scalar('InputFactor{}'.format(i), input_fac) for i, input_fac in enumerate(input_factors)]

    print(excitation0.get_shape().as_list())
    excitation0 = tf.concat([excitation0 * input_fac for input_fac in input_factors], axis=2)
    excitation1 = tf.concat([excitation1 * input_fac for input_fac in input_factors], axis=2)
    return excitation0, excitation1


def read_data(config):
    """
    Reads in the data
    """
    with open(config.data, 'rb') as f:
        train_x, train_y, test_x, test_y = pickle.load(f)
    return test_x, test_y, train_x, train_y


if __name__ == "__main__":
    config = ExperimentConfig(parse_config_args())

    project_folder = os.path.dirname(os.path.realpath(__file__))

    train(config)
