#!/usr/bin/env bash

cd $HOME/lateral-line

python3 ./generate_data.py --force $*
python3 ./sweep.py $*