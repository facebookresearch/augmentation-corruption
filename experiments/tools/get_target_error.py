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
from itertools import combinations
from itertools import product
from scipy.special import comb
import math

parser = argparse.ArgumentParser(description="Collect run summaries.")
parser.add_argument('--cifar10c_dir', dest='baseline_dir', required=True)
parser.add_argument('--log_name', dest='log_name', default='severity_scan.log')

def get_data(base_dirs, exclusions=[], log_file='severity_scan.log'):
    features = {}
    errors = {}
    feature_file = 'features.npz'
    path_stack = base_dirs
    while path_stack:
        curr_dir = path_stack.pop()
        subdirs = [os.path.join(curr_dir, x) for x in os.listdir(curr_dir) if x[0] != '.']
        subdirs = [x for x in subdirs if os.path.isdir(x)]
        path_stack.extend(subdirs)
        summary_file = os.path.join(curr_dir, log_file)
        if os.path.exists(summary_file):
            curr_features = np.load(os.path.join(curr_dir, feature_file))
            features.update({k : v for k,v in curr_features.items() if k.split("-")[0] not in exclusions})
            stats = lu.load_json_stats(summary_file)
            curr_errs = {stats[i]["_type"] : stats[i]["top1_err"] for i in range(len(stats))\
                    if stats[i]["_type"] != "test_epoch" and stats[i]["_type"].split("-")[0] not in exclusions}
            errors.update(curr_errs)
    return errors, features

def get_target_error(baseline_errs):
    errs = [err for b, err in baseline_errs.items()]
    return sum(errs)/len(errs)



def dict_avg(list_of_dicts):
    out = None
    for d in list_of_dicts:
        if out is None:
            out = d
        else:
            for k in out:
                out[k] += d[k]
    for k in out:
        out[k] /= len(list_of_dicts)

    std = None
    for d in list_of_dicts:
        if std is None:
            std = {}
            for k in out:
                std[k] = (d[k]-out[k])**2
        else:
            for k in out:
                std[k] += (d[k]-out[k])**2
    for k in std:
        std[k] = math.sqrt(std[k]) / len(list_of_dicts)


    return out, std




def main():
    args = parser.parse_args()
    baseline_dir = args.baseline_dir
    log_name = args.log_name
    baseline_exclusions = ['saturate', 'spatter', 'gaussian_blur', 'speckle_noise']

    print("Loading data...")
    baseline_dirs = baseline_dir.split(",")
    baseline_errs_list = []
    baseline_features_list = []
    for baseline_dir in baseline_dirs:
        baseline_errs, baseline_features = get_data([baseline_dir], log_file=log_name, exclusions=baseline_exclusions)
        baseline_errs_list.append(baseline_errs)
        baseline_features_list.append(baseline_features)

    baseline_errs, baseline_std = dict_avg(baseline_errs_list)


    target_error = get_target_error(baseline_errs)
    print(target_error)

if __name__=="__main__":
    main()
