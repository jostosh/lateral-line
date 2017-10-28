import argparse
import os

from latline.util.parameter import Parameter


class Config:
    version = 'v0.0'

    log_base = os.path.join(os.getcwd(), 'logs')
    log_sub = ''
    log_dir = None

    @classmethod
    def load(cls):
        parser = argparse.ArgumentParser()
        custom_parsers = {}
        for arg, default in cls.__dict__.items():
            if (callable(default) and arg == 'load') or (len(arg) > 2 and '__' == arg[2:] == arg[-2:]):
                continue
            if isinstance(default, bool):
                parser.add_argument("--" + arg, action='store_true', dest=arg)
                parser.add_argument("--not_" + arg, action='store_false', dest=arg)
                parser.set_defaults(**{arg: default})
            else:
                if isinstance(default, Parameter):
                    # Parameter offers a few more options
                    parser.add_argument("--" + arg, **default.options)
                    if default.has_parser():
                        custom_parsers[arg] = default
                else:
                    # Default usage, only a default value is given of which the type is inferred
                    parser.add_argument("--" + arg, type=type(default), default=default)

        args = parser.parse_args()

        for key, val in vars(args).items():
            if key in custom_parsers:
                setattr(cls, key, custom_parsers[key].parse(val))
            else:
                setattr(cls, key, val)

        cls.log_dir = _get_log_dir(Config)


def _get_log_dir(config: Config, base=None):
    path = base or os.path.join(config.log_base, config.version, config.log_sub)
    os.makedirs(path, exist_ok=True)
    # Check the current directories in there
    current_dirs = sorted([o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))])
    current_dirs = list(filter(lambda s: 'run' in s, current_dirs))
    # We're assuming at this point that we do not exceed 1M runs per version
    if not current_dirs:
        # If there are no dirs yet, start at 0
        rundir = 'run000000'
    else:
        # Otherwise make a new one by incrementing the count
        lastdir     = current_dirs[-1]
        lastrun     = int(lastdir[3:])
        rundir      = "run%06d" % (lastrun + 1,)
    fulldir = os.path.join(path, rundir)
    return fulldir

