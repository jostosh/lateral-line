from argparse import ArgumentParser

data_config0 = {
    "v": 0.5,
    "x_range": [-1, 1],
    "y_range": [-1, 1],
    "z_range": [0, 1.5],
    "d_theta_range": [-.5, .5],
    "sensor_range": [-1.5, 1.5],
    "z_res": 24,
    "N_train": 10000,
    "N_test": 2000,
    "N_val" : 2000,
    "N_sensors": 32,
    "display": False,
    "sigma": 0.2,
    "tau": 2
}


def parse_data_config_args():
    """
    This function sets the command line arguments to look for. The defaults are given in config1 above.
    :return:
    """
    parser = ArgumentParser()
    for name, val in data_config0.items():
        if isinstance(val, bool):
            parser.add_argument('--' + name, action='store_true', dest=name)
            parser.add_argument('--not_' + name, action='store_false', dest=name)
            parser.set_defaults(**{name: val})
        else:
            parser.add_argument('--' + name, type=type(val), default=val)

    args = parser.parse_args()
    return args


class DataConfig(object):
    def __init__(self, args):
        if isinstance(args, dict):
            self.__dict__.update(**args)
            return

        self.v = args.v
        self.x_range = args.x_range
        self.y_range = args.y_range
        self.z_range = args.z_range
        self.d_theta_range = args.d_theta_range
        self.z_res = args.z_res
        self.N_train = args.N_train
        self.N_test = args.N_test
        self.N_val = args.N_val
        self.N_sensors = args.N_sensors
        self.display = args.display
        self.sigma = args.sigma
        self.sensor_range = args.sensor_range
