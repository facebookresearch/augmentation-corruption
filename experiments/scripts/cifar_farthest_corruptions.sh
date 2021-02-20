#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# cifar_dataset_distances.sh should be run 10 times, and the $RUN_<i>_DIR
# variable set for each.
CWD=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CWD/..
set -u

export D1=$RUN_1_DIR/datasets.csv
export D2=$RUN_2_DIR/datasets.csv
export D3=$RUN_3_DIR/datasets.csv
export D4=$RUN_4_DIR/datasets.csv
export D5=$RUN_5_DIR/datasets.csv
export D6=$RUN_6_DIR/datasets.csv
export D7=$RUN_7_DIR/datasets.csv
export D8=$RUN_8_DIR/datasets.csv
export D9=$RUN_9_DIR/datasets.csv
export D10=$RUN_10_DIR/datasets.csv
export TARGET_ERROR=26.9052133333333 # Average of 10 runs tested on CIFAR10-C

python ../calc_distance_shifts.py --input_files=$D1,$D2,$D3,$D4,$D5,$D6,$D7,$D8,$D9,$D10 --target=$TARGET_ERROR
