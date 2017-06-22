from latline.experiment_config import ExperimentConfig, parse_config_args
from train import train
import pprint
import tensorflow as tf
from argparse import ArgumentParser


sweeps = {
    0: dict(model='parallel', merge_at=4, logdir_base_suffix='parallel_default'),
    1: dict(model='parallel', merge_at=3, logdir_base_suffix='parallel_m3'),
    2: dict(model='parallel', merge_at=2, logdir_base_suffix='parallel_m2'),
    3: dict(model='cross', logdir_base_suffix='cross_default'),
    4: dict(model='parallel', logdir_base_suffix='parallel_multi_range', multi_range=True),
    5: dict(model='parallel', logdir_base_suffix='parallel_multi_range_tr', multi_range=True, multi_range_trainable=True),
    6: dict(model='parallel', logdir_base_suffix='parallel_multi_range_tr', multi_range=True, multi_range_trainable=True),
    7: dict(model='parallel', merge_at=3, logdir_base_suffix='parallel_dense'),
    8: dict(model='parallel', merge_at=3),
    9: dict(model='parallel', merge_at=1, logdir_base_suffix='parallel_m1'),
    10: dict(model='parallel', merge_at=0, logdir_base_suffix='parallel_m0'),
    11: dict(model='parallel', noise=0.1, logdir_base_suffix='parallel_n01'),
    12: dict(model='parallel', noise=0.1, logdir_base_suffix='parallel_n02')
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--until", type=int, default=len(sweeps))
    parser.add_argument("--subset", type=int, default=[], nargs='+')
    args = parser.parse_args()
    config = ExperimentConfig()

    sweep_indices = range(args.start, args.until) if not args.subset else args.subset

    for i in sweep_indices:
        config.__dict__.update(sweeps[i])
        print("Starting new parameter sweep")
        pprint.pprint(config.__dict__)
        tf.reset_default_graph()
        train(config)
