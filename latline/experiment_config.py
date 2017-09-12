from argparse import ArgumentParser
import os
import pprint

PROJECT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DEFAULT_ACTIVATIONS = ['tanh', 'relu', 'relu', 'relu', 'sigmoid']
VERSION = 'v0.5'

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
    'sensitivity': 1000,
    'fnm': 'multisphere',
    'force': False
}

experiment_config0 = {
    "n_kernels": [32, 64, 64, 64],
    "n_units": [1024],
    "batch_size": 64,
    "kernel_shapes": [[5, 7], [5, 7], [5, 7], [5, 7], 5],
    "half_time": 1000,
    "data": os.path.join(PROJECT_FOLDER, 'data'),
    "activations": DEFAULT_ACTIVATIONS,
    "logdir": os.path.join(PROJECT_FOLDER, "tensorboardlogs"),
    "n_epochs": 250,
    "merge_at": 3,
    "model": "parallel",
    "output_layer": len(DEFAULT_ACTIVATIONS) - 1,
    "loss": "cross_entropy",
    "lr": 1e-3,
    'n_sensors': 32,
    'fc_activations': ['relu', 'sigmoid'],
    'dense': False,
    'multi_range': False,
    'multi_range_trainable': False,
    'input_factors': [0.1, 1.0, 10.0],
    'noise': 0.01,
    'logdir_base_suffix': 'test',
    'optimizer': 'adam',
    'fnm': 'multisphere',
    'tau': 2
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
    def __init__(self, args=None):
        if args is None:
            self.__dict__.update(**experiment_config0)
            return
        if isinstance(args, dict):
            self.__dict__.update(**args)
            return
        self.n_kernels = args.n_kernels
        self.batch_size = args.batch_size
        self.kernel_shapes = args.kernel_shapes
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
        self.dense = args.dense
        self.multi_range = args.multi_range
        self.multi_range_trainable = args.multi_range_trainable
        self.input_factors = args.input_factors
        self.noise = args.noise
        self.logdir_base_suffix = args.logdir_base_suffix
        self.optimizer = args.optimizer
        self.fnm = args.fnm
        self.tau = args.tau


class DataConfig(object):
    """
    This object forces code completion in IDE, which can be very convenient in Python
    """
    def __init__(self, args=None):
        if args is None:
            self.__dict__.update(**data_config0)
            return
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
        self.fnm = args.fnm
        self.force = args.force


def init_log_dir(config, by_params=[]):
    """
    Automatically creates a logging dir for TensorBoard logging
    :param config:      ExperimentConfig object
    :param by_params:   List of params to use in the generation of the particular TensorBoard logging directory
    :return:            The newly created logging dir
    """
    base = os.path.join(config.logdir, VERSION, config.logdir_base_suffix)
    if by_params:
        base = os.path.join(base, *['{}={}'.format(p, config.__dict__[p]) for p in by_params])

    os.makedirs(base, exist_ok=True)
    dirs = os.listdir(base)

    logdir = os.path.join(base, 'run%04d' % (int(sorted(dirs)[-1][-4:]) + 1,) if dirs else 'run0000')
    os.makedirs(logdir)

    with open(os.path.join(logdir, 'config.txt'), 'w') as f:
        pprint.pprint(config.__dict__, stream=f)

    return logdir
