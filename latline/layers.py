import tensorflow as tf
from functools import partial


def get_activation(name):
    """
    Returns an activation function by the name
    :param name:    The name of the function
    :return:        The activation function
    """
    return {
        'relu': tf.nn.relu,
        'sigmoid': tf.nn.sigmoid,
        'tanh': tf.nn.tanh
    }[name]


def fully_connected(incoming, n_units, activation='relu', name='FullyConnected'):
    """
    Defines a conv1d using default settings
    :param incoming:        Incoming Tensor
    :param n_filters:       The number of output features
    :param filter_shape:    The filter shape (integer)
    :param activation:      The activation function name
    :param name:            The name of the layer.
    :return:                Returns the output tensor.
    """
    n_in = incoming.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0) if activation == 'relu' \
            else tf.contrib.layers.xavier_initializer()
        weights = tf.get_variable("weights", initializer=initializer,
                                  shape=[n_in, n_units],
                                  regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        bias = tf.get_variable("biases", initializer=tf.constant_initializer(), shape=[n_units],
                               regularizer=tf.contrib.layers.l2_regularizer(1e-4))

    f = get_activation(activation)

    activation = f(tf.nn.xw_plus_b(incoming, weights, bias), name=name+'Out')
    summary = tf.summary.histogram(name + 'Out', activation)
    tf.add_to_collection(tf.GraphKeys.SUMMARIES + "/train", summary)
    tf.add_to_collection(tf.GraphKeys.SUMMARIES + "/test", summary)

    return activation


def conv1d(incoming, n_filters, filter_shape, activation='relu', name='Conv1D'):
    """
    Defines a conv1d using default settings
    :param incoming:        Incoming Tensor
    :param n_filters:       The number of output features
    :param filter_shape:    The filter shape (integer)
    :param activation:      The activation function name
    :param name:            The name of the layer.
    :return:                Returns the output tensor.
    """
    if isinstance(filter_shape, int):
        filter_shape = [filter_shape]

    n_filters //= len(filter_shape)
    n_in = incoming.get_shape().as_list()[-1]

    outputs = []
    with tf.variable_scope(name):
        for i, f_s in enumerate(filter_shape):
            initializer = tf.contrib.layers.xavier_initializer_conv2d()
            weights = tf.get_variable("weights{}".format(i), initializer=initializer,
                                      shape=[f_s, n_in, n_filters],
                                      regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            bias = tf.get_variable("biases{}".format(i), initializer=tf.constant_initializer(), shape=[n_filters],
                                   regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            f = get_activation(activation)
            outputs.append(
                f(tf.nn.bias_add(tf.nn.conv1d(incoming, weights, stride=1, padding='SAME'), bias), name=name + 'Out')
            )

    out = tf.concat(outputs, axis=2)

    summary = tf.summary.histogram(name + 'Out', out)
    tf.add_to_collection(tf.GraphKeys.SUMMARIES + "/train", summary)
    tf.add_to_collection(tf.GraphKeys.SUMMARIES + "/test", summary)

    return out


def conv_chain(incoming, n_kernels, kernel_shapes, activations, count_from=0, dense=False):
    """
    Builds a chain of convolution operations by looping over the set over the list of kernel counts, shapes and
    activation functions
    :param incoming:        The incoming Tensor
    :param n_kernels:       A list of ints defining the number of kernels to use per layer
    :param kernel_shapes:   The kernel shapes defining the width of each kernel
    :param activations:     The activation function to use at each layer
    :param count_from:      Used for variable scope naming: e.g. if count_from = 4 and there are 2 conv layers being
                            created, they will use conv4 and conv5
    :return:                The final Tensor in the chain
    """
    chain = incoming
    prev_layers = []
    for i, (n_k, kernel_shape, activation) in enumerate(zip(n_kernels, kernel_shapes, activations)):
        chain_input = chain if not dense else tf.concat(prev_layers, axis=2)
        chain = conv1d(chain_input, n_k, kernel_shape, activation=activation, name='Conv{}'.format(i + count_from))
        prev_layers.append(chain)
    return chain


def fully_connected_chain(incoming, n_units, activations, count_from=0):
    """
    Builds a chain of convolution operations by looping over the set over the list of kernel counts, shapes and
    activation functions
    :param incoming:        The incoming Tensor
    :param n_kernels:       A list of ints defining the number of kernels to use per layer
    :param kernel_shapes:   The kernel shapes defining the width of each kernel
    :param activations:     The activation function to use at each layer
    :param count_from:      Used for variable scope naming: e.g. if count_from = 4 and there are 2 conv layers being
                            created, they will use conv4 and conv5
    :return:                The final Tensor in the chain
    """
    chain = tf.contrib.layers.flatten(incoming)
    for i, (n_units, activation) in enumerate(zip(n_units, activations)):
        chain = fully_connected(chain, n_units, activation=activation, name='FullyConnected{}'.format(i + count_from))
    return chain


def binary_cross_entropy_loss(prediction, targets):
    return -tf.reduce_mean(
        tf.reduce_sum(
            targets * tf.log(prediction + 1e-20) + (1 - targets) * tf.log(1 - prediction + 1e-20),
            axis=[1, 2]
        )
    )


def define_multi_range_input(trainable, input_factors, excitation0, excitation1):
    if trainable:
        input_factors = [tf.Variable(input_fac) for input_fac in input_factors]
        [tf.summary.scalar('InputFactor{}'.format(i), input_fac) for i, input_fac in enumerate(input_factors)]

    excitation0 = tf.concat([excitation0 * input_fac for input_fac in input_factors], axis=2)
    excitation1 = tf.concat([excitation1 * input_fac for input_fac in input_factors], axis=2)
    return excitation0, excitation1


def noise_layer(incoming, std):
    """
    Adds noise to input
    :param incoming: Input tensor
    :param std:      Standard deviation
    :return:         Tensor with added Gaussian noise
    """
    noise = tf.random_normal(shape=tf.shape(incoming), mean=0.0, stddev=std, dtype=tf.float32)
    return incoming + noise