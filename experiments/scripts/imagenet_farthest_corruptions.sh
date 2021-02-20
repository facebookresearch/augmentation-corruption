#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# imagenet_dataset_distances.sh should be run 5 times, and the $RUN_<i>_DIR
# variable set for each.
CWD=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CWD/..
set -u

export D1=$RUN_1_DIR/datasets.csv
export D2=$RUN_2_DIR/datasets.csv
export D3=$RUN_3_DIR/datasets.csv
export D4=$RUN_4_DIR/datasets.csv
export D5=$RUN_5_DIR/datasets.csv
export TARGET_ERROR=58.184448 # Average of 5 runs tested on ImageNet-C

python ../calc_distance_shifts.py --input_files=$D1,$D2,$D3,$D4,$D5 --target=$TARGET_ERROR
