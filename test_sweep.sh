#!/usr/bin/env bash

cd $HOME/lateral-line

python3 ./generate_data.py $*
python3 ./sweep.py $*