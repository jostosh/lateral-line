import os
import pprint
from latline.util.config import Config
from latline.util.parameter import Parameter, LiteralParser


PROJECT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DEFAULT_ACTIVATIONS = ['tanh', 'relu', 'relu', 'relu', 'linear']
VERSION = 'v0.6'


class DataConfig(Config):
    v = 0.05
    x_range = Parameter(default=[-1, 1], nargs=2, type=float)
    y_range = Parameter(default=[-1, 1], nargs=2, type=float)
    z_range = Parameter(default=[0, 2.0], nargs=2, type=float)
    d_theta_range = Parameter(default=[-0.5, 0.5], nargs=2, type=float)
    sensor_range = Parameter(default=[-1.5, 1.5], nargs=2, type=float)
    resolution = 64
    N_train = 10000
    N_test = 2000
    N_val = 2000
    n_sensors = 128
    display = False
    sigma = 0.2
    tau = 4
    r = 0.05
    min_spheres = 1
    max_spheres = 2
    sensitivity = 1000
    fnm = 'multisphere'
    force = False


class ExperimentConfig(Config):
    log_base = os.path.join(os.getcwd(), 'logs')
    log_sub = ''
    log_dir = None
    version = VERSION
    n_kernels = Parameter(default=[32, 64, 64, 64], nargs='+', type=int)
    n_units = Parameter(default=[256], nargs='+', type=int)
    batch_size = 64
    kernel_shapes = Parameter(default='[[5, 7], [5, 7], [5, 7], [5, 7], 5]', parser=LiteralParser())
    half_time = 1000
    activations = Parameter(
        default=['tanh', 'relu', 'relu', 'relu', 'linear'],
        nargs='+', choices=['tanh', 'relu', 'sigmoid', 'linear']
    )
    data = os.path.join(PROJECT_FOLDER, 'data')
    n_epochs = 250
    merge_at = 3
    model = 'parallel'
    output_layer = len(DEFAULT_ACTIVATIONS) - 1
    loss = 'cross_ent_logits'
    lr = 1e-3
    n_sensors = 128
    fc_activations = Parameter(
        default=['relu', 'linear'], nargs='+', type=str, choices=['relu', 'sigmoid', 'linear'])
    dense = False
    multi_range = False
    multi_range_trainable = False
    input_factors = Parameter(default=[0.1, 1.0, 10.0], nargs='+', type=float)
    noise = 0.01
    log_sub = 'test'
    optimizer = 'adam'
    fnm = 'multisphere'
    tau = 4
    resolution = 84
    share = True
    localization_threshold = 0.75


def init_log_dir(config, by_params=[]):
    """
    Automatically creates a logging dir for TensorBoard logging
    :param config:      ExperimentConfig object
    :param by_params:   List of params to use in the generation of the particular TensorBoard
        logging directory
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
