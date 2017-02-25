import os
import pickle

import numpy as np
import tensorflow as tf
from tqdm import trange

from data_util import DataBatcher
from latline.experiment_config import ExperimentConfig, parse_config_args, init_log_dir
from latline.layers import conv_chain


def train(config):
    # Create TensorFlow session
    sess = tf.Session()

    # Read the data from storage
    test_x, test_y, train_x, train_y = read_data(config)

    # Set up two data batchers that provide a straightforward interface for moving through data batches
    data_batcher_train = DataBatcher(train_x, train_y, config.batch_size)
    data_batcher_test = DataBatcher(test_x, test_y, len(test_x))

    # We take the output depth from the target data
    output_depth = train_y.shape[-1] * train_y.shape[1]
    # This should be added to our list of n_kernels
    n_kernels = config.n_kernels + [output_depth]

    # Now we can set up the network
    # First we define the input placeholders
    excitation0, excitation1 = define_inputs((None,) + train_x.shape[-2:])

    # Then, we define the network giving us the output
    out = define_network(config, excitation0, excitation1, n_kernels)

    # Next, we define, the loss that depends on the error + some L2 regularization of the weights
    # For reporting performance, the MSE is used
    loss, mse, target = define_loss(out, train_y)

    # We define the train step which is the Op that should be called when training.
    train_step = define_train_step(loss)

    # Initialize writer for TensorBoard logging
    writer = tf.summary.FileWriter(init_log_dir(config.logdir), sess.graph)

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
            mean_square_error, l, _ = sess.run(
                [mse, loss, train_step],
                feed_dict={excitation0: batch_x[:, 0, :, :], excitation1: batch_x[:, 1, :, :], target: batch_y})
            t.set_postfix(MSE=mean_square_error, loss=l)

        test_x, test_y = data_batcher_test.next_batch()
        print("Test MSE: {}".format(
            sess.run(mse, feed_dict={excitation0: test_x[:, 0, :, :], excitation1: test_x[:, 1, :, :],
                                      target: test_y})))


def read_data(config):
    """
    Reads in the data
    """
    with open(config.data, 'rb') as f:
        train_x, train_y, test_x, test_y, val_x, val_y = pickle.load(f)
    return test_x, test_y, train_x, train_y


def define_train_step(loss):
    """
    Define the train step
    """
    with tf.name_scope("TrainStep"):
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    return train_step


def define_loss(out, train_y):
    """
    Defines the loss plus performance statistics and returns target placeholder
    :param out:
    :param train_y:
    :return:
    """
    with tf.name_scope("Target"):
        shape = train_y.shape
        target = tf.placeholder(tf.float32, (None,) + shape[1:], name="Target")
        target_t = tf.transpose(target, [0, 2, 3, 1])
        target_merged = tf.reshape(target_t, (-1, shape[2], shape[1] * shape[3]))
    with tf.name_scope("Loss"):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.nn.l2_loss(target_merged - out) + tf.add_n(reg_losses)
        mse = tf.reduce_mean(tf.square(target_merged - out))
    return loss, mse, target


def define_network(config, excitation0, excitation1, n_kernels):
    """
    Builds the body of the network
    :param config: An ExperimentConfig instance.
    :param excitation0: First excitation placeholder
    :param excitation1: Second excitation placeholder
    :param n_kernels:   Number of kernels
    :return:
    """

    with tf.variable_scope("SharedStreams") as scope:
        # Build the first chain
        chain0 = conv_chain(excitation0, n_kernels[:config.merge_at], config.kernel_shapes[:config.merge_at],
                            config.activations[:config.merge_at])
        # Reuse the variables of this chain
        scope.reuse_variables()
        # And define the second chain
        chain1 = conv_chain(excitation1, n_kernels[:config.merge_at], config.kernel_shapes[:config.merge_at],
                            config.activations[:config.merge_at])
    with tf.name_scope("MergedStream"):
        # Now we merge these to define the input for the last few layers
        convs_concat = tf.concat(2, [chain0, chain1], name='ConvsConcat')
    with tf.variable_scope("Output"):
        # Finally, we define the output of the network using the layer parameters for those after the merge
        out = conv_chain(convs_concat, n_kernels[config.merge_at:], config.kernel_shapes[config.merge_at:],
                         config.activations[config.merge_at:], count_from=config.merge_at)
    return out


def define_inputs(shape):
    """
    Define the inputs
    :param train_x: The numeric train input that is used
    :return: Two placeholders for both sensor arrays
    """
    with tf.name_scope("Inputs"):
        excitation0 = tf.placeholder(tf.float32, shape=shape, name="Excitation0")
        excitation1 = tf.placeholder(tf.float32, shape=shape, name="Excitation1")
    return excitation0, excitation1


if __name__ == "__main__":
    config = ExperimentConfig(parse_config_args())

    project_folder = os.path.dirname(os.path.realpath(__file__))

    train(config)
