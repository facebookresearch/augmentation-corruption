#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Set $MODEL_WEIGHTS to the checkpoint for a pycls ResNet-50 model.
CWD=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CWD/..
set -u
export LAUNCHER=submitit_local

python ../test_imagenet.py weights=$MODEL_WEIGHTS corrupt.aug_string=caustic_refraction-2.35,caustic_refraction-3.2,caustic_refraction-4.9,caustic_refraction-6.6,caustic_refraction-9.15,inverse_sparkles-1.0,inverse_sparkles-2.0,inverse_sparkles-4.0,inverse_sparkles-9.0,inverse_sparkles-10.0,sparkles-1.0,sparkles-2.0,sparkles-3.0,sparkles-5.0,sparkles-6.0,perlin_noise-4.6,perlin_noise-5.2,perlin_noise-5.8,perlin_noise-7.6,perlin_noise-8.8,blue_noise_sample-0.8--plasma_noise-4.75--checkerboard_cutout-2.0--cocentric_sine_waves-3.0--single_frequency_greyscale-1.0--brownish_noise-1.0,blue_noise_sample-1.6--plasma_noise-7.0--checkerboard_cutout-3.0--cocentric_sine_waves-5.0--single_frequency_greyscale-1.5--brownish_noise-2.0,blue_noise_sample-2.4--plasma_noise-8.5--checkerboard_cutout-4.0--cocentric_sine_waves-8.0--single_frequency_greyscale-2.0--brownish_noise-3.0,blue_noise_sample-4.0--plasma_noise-9.25--checkerboard_cutout-5.0--cocentric_sine_waves-9.0--single_frequency_greyscale-4.5--brownish_noise-4.0,blue_noise_sample-5.6--plasma_noise-10.0--checkerboard_cutout-6.0--cocentric_sine_waves-10.0--single_frequency_greyscale-5.0--brownish_noise-5.0 rng_seed=$RANDOM corrupt.mCE_baseline_file= hydra/launcher=$LAUNCHER -m
