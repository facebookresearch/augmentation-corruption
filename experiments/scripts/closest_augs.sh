#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Set $FEATURE_DIR to the place to save features and sorted transforms
CWD=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CWD/..
set -u
export LAUNCHER=submitit_local

# Just train a feature extractor and generate features, so training on those
# features is unnecessary (hence optim.max_epoch=1).
python ../closest_augs.py rng_seed=$RANDOM optim.max_epoch=1 num_transforms=100000 hydra.sweep.dir=$FEATURE_DIR hydra.sweep.subdir=. hydra/launcher=$LAUNCHER -m

python ../closest_augs.py transform_file=$FEATURE_DIR/sorted_transforms.npy num_transforms=1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000,60000,70000,80000,90000,97000,99000,99700,99900,99970,99990 selection_type=closest,farthest,random rng_seed='${hydra:job.num}' hydra/launcher=$LAUNCHER -m
