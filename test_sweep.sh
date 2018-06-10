#!/usr/bin/env bash

set -e

for i in `seq 0 4`;
do
        python3 $HOME/lateral-line/generate_data.py --fnm sweep${i}
        python3 $HOME/lateral-line/sweep.py --fnm sweep${i} $*
done
