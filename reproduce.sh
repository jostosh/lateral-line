#!/usr/bin/env bash

set -e

conda env create -f latline.yaml -n latline

source activate latline

bash test_sweep.sh
