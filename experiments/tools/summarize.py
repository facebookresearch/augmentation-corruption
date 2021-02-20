# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import overlap.utils.logging as lu
import decimal
import simplejson
import numpy as np
import omegaconf

parser = argparse.ArgumentParser(description="Collect run summaries.")
parser.add_argument('--dir', dest='run_dir')
parser.add_argument('--filename', dest='summary_name', default='train_cifar10.log')

def main():
    args = parser.parse_args()
    run_dir = args.run_dir
    summary_name = args.summary_name
    hydra_config = '.hydra/config.yaml'

    files = []
    path_stack = [run_dir]
    while path_stack:
        curr_dir = path_stack.pop()
        subdirs = [os.path.join(curr_dir, x) for x in os.listdir(curr_dir) if x[0] != '.']
        subdirs = [x for x in subdirs if os.path.isdir(x)]
        path_stack.extend(subdirs)
        summary_file = os.path.join(curr_dir, summary_name)
        if os.path.exists(summary_file):
            config_file = os.path.join(curr_dir, hydra_config)
            files.append((summary_file, config_file))

    for (summary, config) in files:
        data = []
        cfg = omegaconf.OmegaConf.load(config)
        stats = lu.load_json_stats(summary)

        #Run meta-deta
        data.append(cfg.rng_seed) # ID
        data.append(cfg.name)     # Name
        data.append(summary)      # Filename for data

        #Model info
        data.append(cfg.model['class'].split('.')[-1])  # Model name
        data.append(cfg.model.params.depth)             # Model depth
        data.append(cfg.model.params.widen_factor)      # Width factor

        # Optimizer info
        data.append(cfg.optim.lr_policy['class'].split('.')[-1]) # LR policy
        data.append(cfg.optim.base_lr)                           # Base LR
        data.append(cfg.optim.max_epoch)                         # Num epochs

        # Augmentation info
        aug_data = []
        train_class = cfg.train['class'].split('.')[-1]
        if train_class == 'Cifar10Base': # No augmentation
            aug_data.append('none')
        elif train_class == 'Cifar10Augmix':              # Augmix
            aug_data.append('augmix')
            aug_string = cfg.train.params.aug_string
            if aug_string is None:
                aug_string = 'standard'
            aug_data.append(aug_string)                   # Aug string            
            aug_data.append(cfg.train.params.width)       # Augmix width
            reported_depth = cfg.train.params.depth * (-1 if cfg.train.params.random_depth else -1)
            aug_data.append(reported_depth)               # Augmix depth
            aug_data.append(cfg.train.params.prob_coeff)  # Augmix prob coefficient
            severity = cfg.train.params.severity if cfg.train.params.aug_string is None else ''
            aug_data.append(severity)                     # Augmix severity
        elif train_class == 'Cifar10Corruption':          # Corruption Trained
            aug_data.append('corruption_trained')
            aug_string = cfg.train.params.aug_string
            if aug_string is None:
                aug_string = 'extra' if cfg.train.params.include_extra else 'standard'
            aug_data.append(aug_string)                   # 
        aug_data.extend(['' for i in range(6-len(aug_data))])
        data.extend(aug_data)

        # Feature extraction info
        data.extend(['' for i in range(17)])

        # JSD info
        data.append('no')
        data.extend(['',''])

        # Batch size info
        data.append(cfg.train.batch_size)
        data.append(cfg.test.batch_size)

        # Errors
        clean_error = lu.parse_json_stats(stats, 'test_epoch', 'top1_err')[0]
        data.append(clean_error)  # Clean error
        data.extend(['', ''])           # Space for clean error std and C95
        mCE = lu.parse_json_stats(stats, 'overall-avg', 'mCE')
        mCE = mCE[0] if mCE else ''
        data.append(mCE)          # mCE
        data.extend(['', ''])           # Space for mCE std and C95
        rmCE = lu.parse_json_stats(stats, 'overall-avg', 'rmCE')
        rmCE = rmCE[0] if rmCE else ''
        data.append(rmCE)         # rmCE
        data.extend(['', ''])           # Space for rmCE std and C95
        avg_error = lu.parse_json_stats(stats, 'overall-avg', 'top1_err')[0]
        data.append(avg_error)    # Average corruption error
        data.extend(['', ''])     # Space for corruption error std and C95
        data.extend(['', ''])           # Space for number in average and type of average
        data.append('')           # Divider

        # Individual errors

        # Noise
        data.append(lu.parse_json_stats(stats, 'noise-avg', 'top1_err')[0])

        data.append(lu.parse_json_stats(stats, 'gaussian_noise-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'impulse_noise-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'shot_noise-avg', 'top1_err')[0])

        # Blur
        data.append(lu.parse_json_stats(stats, 'blur-avg', 'top1_err')[0])

        data.append(lu.parse_json_stats(stats, 'defocus_blur-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'glass_blur-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'motion_blur-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'zoom_blur-avg', 'top1_err')[0])

        # Weather
        data.append(lu.parse_json_stats(stats, 'weather-avg', 'top1_err')[0])

        data.append(lu.parse_json_stats(stats, 'brightness-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'fog-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'frost-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'snow-avg', 'top1_err')[0])

        # Digital
        data.append(lu.parse_json_stats(stats, 'digital-avg', 'top1_err')[0])

        data.append(lu.parse_json_stats(stats, 'contrast-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'elastic_transform-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'jpeg_compression-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'pixelate-avg', 'top1_err')[0])

        # Extra
        data.append(lu.parse_json_stats(stats, 'extra-avg', 'top1_err')[0])

        data.append(lu.parse_json_stats(stats, 'gaussian_blur-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'saturate-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'spatter-avg', 'top1_err')[0])
        data.append(lu.parse_json_stats(stats, 'speckle_noise-avg', 'top1_err')[0])

        data = [str(i) for i in data]
        print(",".join(data))


            



if __name__ == "__main__":
    main()
