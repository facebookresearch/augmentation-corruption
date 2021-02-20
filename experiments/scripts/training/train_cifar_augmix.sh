#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Set $MODEL_WEIGHTS to the checkpoint for a pycls ResNet-50 model.
CWD=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CWD/../..
set -u
export LAUNCHER=cifar10

python ../../train_cifar10_jsd.py hydra/launcher=$LAUNCHER rng_seed=$RANDOM -m
