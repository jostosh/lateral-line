import tensorflow as tf


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
    n_in = incoming.get_shape().as_list()[-1]
    with tf.variable_scope(name):

        weights = tf.get_variable("weights", initializer=tf.contrib.layers.xavier_initializer(),
                                  shape=[filter_shape, n_in, n_filters],
                                  regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        bias = tf.get_variable("biases", initializer=tf.constant_initializer(), shape=[n_filters],
                               regularizer=tf.contrib.layers.l2_regularizer(1e-4))

    f = get_activation(activation)
    activation = f(tf.nn.bias_add(tf.nn.conv1d(incoming, weights, stride=1, padding='SAME'), bias), name=name + 'Out')

    summary = tf.summary.histogram(name + 'Out', activation)
    tf.add_to_collection(tf.GraphKeys.SUMMARIES + "/train", summary)
    tf.add_to_collection(tf.GraphKeys.SUMMARIES + "/test", summary)

    return activation


def conv_chain(incoming, n_kernels, kernel_shapes, activations, count_from=0):
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
    for i, (n_k, kernel_shape, activation) in enumerate(zip(n_kernels, kernel_shapes, activations)):
        chain = conv1d(chain, n_k, kernel_shape, activation=activation, name='conv{}'.format(i + count_from))
    return chain
