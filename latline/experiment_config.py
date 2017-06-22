from argparse import ArgumentParser
import os

PROJECT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DEFAULT_ACTIVATIONS = ['tanh', 'relu', 'relu', 'relu', 'sigmoid']

data_config0 = {
    "v": 0.05,
    "x_range": [-1, 1],
    "y_range": [-1, 1],
    "z_range": [0, 1.5],
    "d_theta_range": [-.5, .5],
    "sensor_range": [-1.5, 1.5],
    "resolution": 24,
    "N_train": 10000,
    "N_test": 2000,
    "N_val" : 2000,
    "N_sensors": 32,
    "display": False,
    "sigma": 0.2,
    "tau": 4,
    "a": 0.05,
    'min_spheres': 1,
    'max_spheres': 2,
    'sensitivity': 1000
}

experiment_config0 = {
    "n_kernels": [32, 64, 64, 64],
    "n_units": [1024],
    "batch_size": 125,
    "filter_shapes": [5, 5, 5, 5, 5],
    "half_time": 1000,
    "data": os.path.join(PROJECT_FOLDER, 'data', 'multisphere_parallel.pickle'),
    "activations": DEFAULT_ACTIVATIONS,
    "logdir": os.path.join(PROJECT_FOLDER, "tensorboardlogs"),
    "n_epochs": 1000,
    "merge_at": 4,
    "model": "parallel",
    "output_layer": len(DEFAULT_ACTIVATIONS) - 1,
    "loss": "cross_entropy",
    "lr": 1e-3,
    'n_sensors': 32,
    'fc_activations': ['relu', 'sigmoid']
}


def parse_config_args(mode='experiment'):
    """
    This function sets the command line arguments to look for. The defaults are given in config1 above.
    :return:
    """
    parser = ArgumentParser()
    for name, val in (data_config0 if mode == 'data' else experiment_config0).items():
        if isinstance(val, bool):
            parser.add_argument('--' + name, action='store_true', dest=name)
            parser.add_argument('--not_' + name, action='store_false', dest=name)
            parser.set_defaults(**{name: val})
        elif isinstance(val, list):
            parser.add_argument('--' + name, nargs='+', type=type(val[0]), default=val)
        else:
            parser.add_argument('--' + name, type=type(val), default=val)

    args = parser.parse_args()
    return args


class ExperimentConfig(object):
    """
    This object forces code completion in an IDE
    """
    def __init__(self, args):
        if isinstance(args, dict):
            self.__dict__.update(**args)
        self.n_kernels = args.n_kernels
        self.batch_size = args.batch_size
        self.kernel_shapes = args.filter_shapes
        self.half_time = args.half_time
        self.data = args.data
        self.activations = args.activations
        self.logdir = args.logdir
        self.n_epochs = args.n_epochs
        self.merge_at = args.merge_at
        self.model = args.model
        self.loss = args.loss
        self.lr = args.lr
        self.n_sensors = args.n_sensors
        self.n_units = args.n_units
        self.fc_activations = args.fc_activations


class DataConfig(object):
    """
    This object forces code completion in IDE, which can be very convenient in Python
    """
    def __init__(self, args):
        if isinstance(args, dict):
            self.__dict__.update(**args)
            return

        self.v = args.v
        self.x_range = args.x_range
        self.y_range = args.y_range
        self.z_range = args.z_range
        self.d_theta_range = args.d_theta_range
        self.resolution = args.resolution
        self.N_train = args.N_train
        self.N_test = args.N_test
        self.N_val = args.N_val
        self.N_sensors = args.N_sensors
        self.display = args.display
        self.sigma = args.sigma
        self.sensor_range = args.sensor_range
        self.tau = args.tau
        self.a = args.a
        self.min_spheres = args.min_spheres
        self.max_spheres = args.max_spheres
        self.sensitivity = args.sensitivity



def init_log_dir(config, by_params=['merge_at', 'loss']):
    """
    Automatically creates a logging dir for TensorBoard logging
    :param config:      ExperimentConfig object
    :param by_params:   List of params to use in the generation of the particular TensorBoard logging directory
    :return:            The newly created logging dir
    """
    base = config.logdir
    if by_params:
        base = os.path.join(base, *['{}={}'.format(p, config.__dict__[p]) for p in by_params])

    os.makedirs(base, exist_ok=True)
    dirs = os.listdir(base)

    logdir = os.path.join(base, 'run%04d' % (int(sorted(dirs)[-1][-4:]) + 1,) if dirs else 'run0000')
    os.makedirs(logdir)

    return logdir
