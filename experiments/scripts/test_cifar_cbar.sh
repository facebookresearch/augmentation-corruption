#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Set $MODEL_WEIGHTS to the checkpoint for a WideResNet-40-2 model.
CWD=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CWD/..
set -u
export LAUNCHER=submitit_local

python ../test_cifar10.py weights=$MODEL_WEIGHTS corrupt.aug_string=blue_noise_sample-0.7--blue_noise_sample-1.4--blue_noise_sample-2.1--blue_noise_sample-3.5--blue_noise_sample-4.2--checkerboard_cutout-1.0--checkerboard_cutout-2.0--checkerboard_cutout-3.0--checkerboard_cutout-4.0--checkerboard_cutout-5.0--inverse_sparkles-1.0--inverse_sparkles-1.5--inverse_sparkles-2.5--inverse_sparkles-4.5--inverse_sparkles-5.0--pinch_and_twirl_v2-5.5--pinch_and_twirl_v2-6.0--pinch_and_twirl_v2-6.5--pinch_and_twirl_v2-8.5--pinch_and_twirl_v2-10.0--ripple-5.5--ripple-6.0--ripple-6.5--ripple-7.5--ripple-8.0--brownish_noise-2.5--brownish_noise-4.0--brownish_noise-5.5--brownish_noise-6.5--brownish_noise-7.0--circular_motion_blur-2.8--circular_motion_blur-3.6--circular_motion_blur-4.4--circular_motion_blur-5.2--circular_motion_blur-6.0--lines-0.5--lines-1.0--lines-1.5--lines-2.0--lines-2.5--sparkles-0.7--sparkles-3.5--sparkles-5.6--sparkles-6.3--sparkles-7.0--transverse_chromatic_abberation-1.0--transverse_chromatic_abberation-2.0--transverse_chromatic_abberation-5.0--transverse_chromatic_abberation-9.0--transverse_chromatic_abberation-10.0 rng_seed=$RANDOM corrupt.mCE_baseline_file= hydra/launcher=$LAUNCHER -m
