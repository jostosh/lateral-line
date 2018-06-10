from latline.experiment_config import ExperimentConfig
from argparse import ArgumentParser
import subprocess


sweeps = [
    dict(model='parallel', merge_at=4, log_sub='parallel_default'),
    dict(model='parallel', merge_at=3, log_sub='parallel_m3'),
    dict(model='parallel', merge_at=2, log_sub='parallel_m2'),
    dict(model='cross', log_sub='cross_default'),
    dict(model='parallel', log_sub='parallel_multi_range', multi_range=True),
    dict(model='parallel', log_sub='parallel_dense', dense=True),
    dict(model='parallel', lr=1e-4, log_sub='l2', loss='l2'),
    dict(model='parallel', merge_at=1, log_sub='parallel_m1'),
    dict(model='parallel', merge_at=0, log_sub='parallel_m0'),
    dict(model='parallel', noise=0.001, log_sub='parallel_n0001'),
    dict(model='parallel', noise=0.1, log_sub='parallel_n01'),
    dict(model='parallel', noise=0.2, log_sub='parallel_n02'),
    dict(model='parallel', merge_at=3, log_sub='tau3', tau=3),
    dict(model='parallel', merge_at=3, log_sub='tau2', tau=2),
    dict(model='parallel', merge_at=3, log_sub='tau1', tau=1),
    dict(model='parallel', log_sub='tau4', tau=4),
    dict(model='parallel', log_sub='best', multi_range=True, tau=1),
    dict(model='parallel', merge_at=5, log_sub='parallel_m5'),
    dict(model='parallel', log_sub='no_sharing', share=False)
]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--until", type=int, default=len(sweeps))
    parser.add_argument("--subset", type=int, default=[], nargs='+')
    parser.add_argument("--fnm", type=str, default='multisphere')
    args = parser.parse_args()
    config = ExperimentConfig()

    sweep_indices = range(args.start, args.until) if not args.subset else args.subset

    for i in sweep_indices:
        config.__dict__.update(sweeps[i])
        command = ['python', './train.py', '--fnm={}'.format(args.fnm)] + \
                  ['--{}={}'.format(k, v) if not isinstance(v, bool) else '--{}'.format(k)
                   for k, v in sweeps[i].items()]
        print("Initiating ", ' '.join(command))

        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        # Grab stdout line by line as it becomes available.  This will loop until
        # p terminates.
        while p.poll() is None:
            print(p.stdout.readline())  # This blocks until it receives a newline.

        # When the subprocess terminates there might be unconsumed output
        # that still needs to be processed.
        print(p.stdout.read())