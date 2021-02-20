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
export NEW_CORR_DIR=$WORKING_DIR/imagenetnewc
export IMAGENETC_DIR=$WORKING_DIR/imagenetc
export SAMPLED_DATASETS=$WORKING_DIR/datasets.csv
export TARGET_ERROR=58.184448 # Average of 5 runs tested on ImageNet-C
export TRAINING_LAUNCHER=submitit_local
export SEVERITY_SCAN_LAUNCHER=submitit_local

python ../train_imagenet.py hydra/launcher=$TRAINING_LAUNCHER rng_seed=$RANDOM hydra.sweep.dir=$MODEL_DIR hydra.sweep.subdir=. -m
python ../train_imagenet.py hydra/launcher=$TRAINING_LAUNCHER rng_seed=$RANDOM hydra.sweep.dir=$FEATURE_EXTRACTOR_DIR hydra.sweep.subdir=. -m
python ../tools/sample_image_indices.py --num=10000 --dataset='imagenet' --out=$INDICES_FILE


python ../severity_scan.py hydra/launcher=$SEVERITY_SCAN_LAUNCHER train.weights=$MODEL_DIR/checkpoints/model_epoch_0100.pyth ft.params.dataset_cfg.weights=$FEATURE_EXTRACTOR_DIR/checkpoints/model_epoch_0100.pyth ft_corrupt.indices_file=$INDICES_FILE aug_string=gaussian_noise--shot_noise--impulse_noise--speckle_noise--motion_blur--defocus_blur--zoom_blur--glass_blur--gaussian_blur--brightness--fog--frost--snow--spatter--contrast--pixelate--jpeg_compression--elastic_transform--saturate severity=1,2,3,4,5 rng_seed='${hydra:job.num}' hydra.sweep.dir=$CIFAR10C_DIR -m

python severity_scan_imagenet.py hydra/launcher=$SEVERITY_SCAN_LAUNCHER train.weights=$MODEL_DIR/checkpoints/model_epoch_0100.pyth ft.params.dataset_cfg.weights=$FEATURE_EXTRACTOR_DIR/checkpoints/model_epoch_0100.pyth aug_string=gaussian_noise--shot_noise--impulse_noise--speckle_noise--motion_blur--defocus_blur--zoom_blur--glass_blur--gaussian_blur--brightness--fog--frost--snow--spatter--contrast--pixelate--jpeg_compression--elastic_transform--saturate severity=1,2,3,4,5 ft_corrupt.indices_file=$INDICES_FILE ft_corrupt.params.num_transforms= rng_seed='${hydra:job.num}' -m

python ../severity_scan_imagenet.py hydra/launcher=$SEVERITY_SCAN_LAUNCHER train.weights=$MODEL_DIR/checkpoints/model_epoch_0100.pyth ft.params.dataset_cfg.weights=$FEATURE_EXTRACTOR_DIR/checkpoints/model_epoch_0100.pyth aug_string=plasma_noise-2.5_10,voronoi_noise-2.5_10,caustic_noise-2.5_10,sparkles-0_10,inverse_sparkles-0_10,scatter-0_5,caustic_refraction-1.5_10,pinch_and_twirl_v2-3.5_10,fish_eye_v2-0_4,water_drop-5_10,ripple-2.5_10,perlin_noise-4_10,single_frequency_greyscale-0_5--cocentric_sine_waves-0_10--checkerboard_cutout-0_10--lines-0_8--chromatic_abberation-0_10--transverse_chromatic_abberation-0_10--bleach_bypass-0_10--technicolor-0_10--pseudocolor-1_10--color_balance-0_10--quadrilateral_no_bars-0_10--perspective_no_bars-0_10--blue_noise_sample-0_8--blue_noise-0_7--hue_shift-0_10--circular_motion_blur-1.5_10--color_dither-2_12--brownish_noise-0_10 severity=1,2,3,4,5,6,7,8,9,10 ft_corrupt.indices_file=$INDICES_FILE ft_corrupt.params.num_transforms= rng_seed='${hydra:job.num}' -m

python ../sample_datasets.py --new_corr_dir=$NEW_CORR_DIR --baseline_corr_dir=$IMAGENETC_DIR --precision 0.01 --num 5 --out=$SAMPLED_DATASETS --target=$TARGET_ERROR
