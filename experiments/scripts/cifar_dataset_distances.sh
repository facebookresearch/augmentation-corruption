#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Set $WORKING_DIR
CWD=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CWD/..
set -u

export MODEL_DIR=$WORKING_DIR/model
export FEATURE_EXTRACTOR_DIR=$WORKING_DIR/ft
export INDICES_FILE=$WORKING_DIR/indices.npy
export NEW_CORR_DIR=$WORKING_DIR/cifar10newc
export CIFAR10C_DIR=$WORKING_DIR/cifar10c
export SAMPLED_DATASETS=$WORKING_DIR/datasets.csv
export TARGET_ERROR=26.9052133333333 # Average of 10 runs tested on CIFAR10-C
export TRAINING_LAUNCHER=submitit_local
export SEVERITY_SCAN_LAUNCHER=submitit_local

python ../train_cifar10.py rng_seed=$RANDOM hydra.sweep.dir=$MODEL_DIR hydra.sweep.subdir=. hydra/launcher=$TRAINING_LAUNCHER optim.max_epoch=1 -m
python ../train_cifar10.py rng_seed=$RANDOM hydra.sweep.dir=$FEATURE_EXTRACTOR_DIR hydra.sweep.subdir=. hydra/launcher=$TRAINING_LAUNCHER optim.max_epoch=1 -m
python ../tools/sample_image_indices.py --num=100 --dataset='cifar-10' --out=$INDICES_FILE

python ../severity_scan.py train.weights=$MODEL_DIR/checkpoints/model_epoch_0100.pyth ft.params.dataset_cfg.weights=$FEATURE_EXTRACTOR_DIR/checkpoints/model_epoch_0100.pyth ft_corrupt.indices_file=$INDICES_FILE aug_string=gaussian_noise--shot_noise--impulse_noise--speckle_noise--motion_blur--defocus_blur--zoom_blur--glass_blur--gaussian_blur--brightness--fog--frost--snow--spatter--contrast--pixelate--jpeg_compression--elastic_transform--saturate severity=1,2,3,4,5 rng_seed='${hydra:job.num}' hydra.sweep.dir=$CIFAR10C_DIR hydra/launcher=$SEVERITY_SCAN_LAUNCHER -m

python ../severity_scan.py train.weights=$MODEL_DIR/checkpoints/model_epoch_0100.pyth ft.params.dataset_cfg.weights=$FEATURE_EXTRACTOR_DIR/checkpoints/model_epoch_0100.pyth ft_corrupt.indices_file=$INDICES_FILE aug_string=single_frequency_greyscale-0_3--cocentric_sine_waves-0_5--plasma_noise-0_7--voronoi_noise-1_5--caustic_noise-2_7--sparkles-0_7--inverse_sparkles-0_5--checkerboard_cutout-0_10--lines-0_5--scatter-1_5--chromatic_abberation-0_7--transverse_chromatic_abberation-0_10--caustic_refraction-1_5--pinch_and_twirl_v2-5_10--fish_eye_v2-0_1--water_drop-2_10--ripple-5_10--bleach_bypass-0_10--technicolor-0_10--pseudocolor-0_7--color_balance-0_10--quadrilateral_no_bars-0_10--perspective_no_bars-0_10--perlin_noise-4_10--blue_noise_sample-0_7--blue_noise-0_2--hue_shift-0_10--circular_motion_blur-2_10--color_dither-0_10--brownish_noise-2_7 severity=1,2,3,4,5,6,7,8,9,10 rng_seed='${hydra:job.num}' hydra.sweep.dir=$NEW_CORR_DIR hydra/launcher=$SEVERITY_SCAN_LAUNCHER -m

python ../sample_datasets.py --new_corr_dir=$NEW_CORR_DIR --baseline_corr_dir=$CIFAR10C_DIR --precision 0.01 --num 5 --out=$SAMPLED_DATASETS --target=$TARGET_ERROR
