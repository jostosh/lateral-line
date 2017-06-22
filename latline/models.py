import tensorflow as tf
import numpy as np

from latline.layers import binary_cross_entropy_loss, conv_chain, fully_connected_chain


def define_train_step(loss, lr):
    """
    Define the train step
    """
    with tf.name_scope("TrainStep"):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
    return train_step, global_step


def define_loss(out, train_y, loss_fn):
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
        # Obtain the regularization losses
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        # The total loss is defined as the L2 loss of the targets together with the L2 losses
        if loss_fn == 'cross_entropy':
            loss = binary_cross_entropy_loss(targets=target_merged, prediction=out)
        elif loss_fn == 'l2':
            loss = tf.nn.l2_loss(target_merged - out)
        else:
            raise ValueError("Unknown loss function")

        loss += tf.add_n(reg_losses)

        # For reporting performance, we use the mean squared error
        mse = tf.reduce_mean(tf.square(target_merged - out))

        # Add some scalar summaries
        loss_summary = tf.summary.scalar("Loss", loss)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES + '/train', loss_summary)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES + '/test', loss_summary)

        mse_summary = tf.summary.scalar("MeanSquaredError", mse)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES + '/train', mse_summary)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES + '/test', mse_summary)

    return loss, mse, target


def define_shared_stream(excitation0, excitation1, n_kernels, kernel_shapes, activations, name="SharedStream",
                         dense=False):
    with tf.variable_scope(name) as scope:

        # Build the first chain
        chain0 = conv_chain(excitation0, n_kernels, kernel_shapes, activations)
        # Reuse the variables of this chain
        scope.reuse_variables()
        # And define the second chain
        chain1 = conv_chain(excitation1, n_kernels, kernel_shapes, activations)

    return chain0, chain1


def define_cross_network(config, excitation0, excitation1):
    """"""

    chain0, chain1 = define_shared_stream(
        excitation0, excitation1, n_kernels=config.n_kernels[:config.merge_at],
        kernel_shapes=config.kernel_shapes[:config.merge_at], activations=config.activations[:config.merge_at]
    )
    with tf.name_scope("MergedStream"):
        convs_concat = tf.concat([chain0, chain1], 2, name='ConvsConcat')
    with tf.name_scope("Tail"):
        out = fully_connected_chain(convs_concat, n_units=config.n_units, activations=config.fc_activations,
                                    count_from=config.merge_at)
        out = tf.reshape(out, [-1, config.n_sensors, config.n_kernels[-1]])

    return out


def define_parallel_network(config, excitation0, excitation1):
    """
    Builds the body of the network
    :param config: An ExperimentConfig instance.
    :param excitation0: First excitation placeholder
    :param excitation1: Second excitation placeholder
    :param n_kernels:   Number of kernels
    :return:
    """

    chain0, chain1 = define_shared_stream(
        excitation0, excitation1, n_kernels=config.n_kernels[:config.merge_at],
        kernel_shapes=config.kernel_shapes[:config.merge_at], activations=config.activations[:config.merge_at],
        dense=config.dense
    )

    #if config.model == Models.PARALLEL:
    with tf.name_scope("MergedStream"):
        # Now we merge these to define the input for the last few layers
        convs_concat = tf.concat([chain0, chain1], 2, name='ConvsConcat')
    with tf.variable_scope("Tail"):
        # Finally, we define the output of the network using the layer parameters for those after the merge
        n_kernels = np.asarray(config.n_kernels[config.merge_at:])
        n_kernels[:-1] *= 2
        out = conv_chain(convs_concat, n_kernels, config.kernel_shapes[config.merge_at:],
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