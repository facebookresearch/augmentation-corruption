# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Calculate corruptions distance "\
        "to the baseline corruptions and find a representative dataset of "\
        "corruptions that are farthest away.")
parser.add_argument('--input_files', type=str, required=True,
        help='Comma separated list of files for error matched corruptions.')
parser.add_argument('--target_error', type=float, required=True,
        help='Target ImageNet-C error for choosing a representative dataset.'
        )

def calc_shifts(corrs_files):
    corr_shifts_all = []
    for corrs_file in corrs_files:
        with open(corrs_file, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]

        # Data is [corruption-string, error, distance]
        data = [line.split(",") for line in lines[1:]]

        corrs = set()
        # Add the corruptions as a set to the data for easy determination of the intersection
        for i in range(len(data)):
            curr_corrs = set([a.split("-")[0] for a in data[i][0].split("--")])
            data[i].append(curr_corrs)
            corrs.update(curr_corrs) # Collect all types of corruptions seen for access later
        corrs = list(corrs)

        # Sample random sets of 10 corruptions
        sampled = []
        while len(sampled) < 100000:
            chosen = np.random.randint(low=0, high=len(data), size=2)
            # Take only disjoint combinations to get a full 10 sampled.
            if not (data[chosen[0]][-1] & data[chosen[1]][-1]):
                sampled.append((
                    "--".join([data[chosen[0]][0], data[chosen[1]][0]]), # Combined aug string
                    (float(data[chosen[0]][1]) + float(data[chosen[1]][1])) / 2, # Average error
                    (float(data[chosen[0]][2]) + float(data[chosen[1]][2])) / 2  # Average distance
                ))

        # Calculate shifts associated with each corruption
        corr_shifts = []
        sampled_mean = np.mean([float(s[2]) for s in sampled]) # Mean error
        sampled_std = np.std([float(s[2]) for s in sampled]) # Mean std
        # Get per corruption shifts in distance
        for corr in corrs:
            pos = []
            # Find all distances from datasets that include this corruption
            for s in sampled:
                included_corrs = [a.split("-")[0] for a in s[0].split("--")]
                if corr in included_corrs:
                    pos.append(float(s[2]))
            # Calculate average shift for this corruption
            pos_avg = np.mean(pos)
            # Shift by average distance and reweight by distance std
            shift = (pos_avg - sampled_mean) / sampled_std
            corr_shifts.append(shift)
        corr_shifts_all.append(corr_shifts)

    # Calculate mean and std across multiple runs
    corr_shifts_all = np.array(corr_shifts_all)
    corr_shifts_mean = np.mean(corr_shifts_all, axis=0)
    corr_shifts_std = np.std(corr_shifts_all, axis=0)
    shifts = {corr : (corr_shifts_mean[i], corr_shifts_std[i]) for i, corr in enumerate(corrs)}
    return shifts

def topk_shifts(shifts, k):
    shifts_list = np.array([v[0] for k, v in shifts.items()])
    corrs_list = np.array([k for k, v in shifts.items()])
    ordered_idx = np.argsort(shifts_list)
    topk = ordered_idx[-k:]
    return corrs_list[topk]

def get_farthest_dataset(farthest_corrs, corrs_files, target_error):
    farthest_corrs = set(farthest_corrs)
    valid_all = []
    for corrs_file in corrs_files:
        valid = []
        with open(corrs_file, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]

        data = [line.split(",") for line in lines[1:]]

        for i in range(len(data)):
            data[i].append(set([a.split("-")[0] for a in data[i][0].split("--")]))
        for datum in data:
            augs = datum[-1]
            if len(augs & farthest_corrs) == 5:
                valid.append(datum)
        valid_all.append(valid)

    matched_all = []
    for valid in valid_all:
        matched = []
        for i in np.arange(len(valid)):
            for j in np.arange(start=i+1, stop=len(valid)):
                if not (valid[i][-1] & valid[j][-1]):
                    matched.append((
                        "--".join([valid[i][0], valid[j][0]]), # Combined corruption string
                        (float(valid[i][1]) + float(valid[j][1])) / 2, # Average error
                        (float(valid[i][2]) + float(valid[j][2])) / 2  # Average distance
                        ))
        matched_all.append(matched)

    best = None
    for i, matched in enumerate(matched_all):
        for m in matched:
            if best is None or np.abs(m[1]-target_error) < np.abs(best[1] - target_error):
                best = m
                best_corr_dir = i

    return best

def main():
    args = parser.parse_args()
    file_list = args.input_files.split(",")
    shifts = calc_shifts(file_list)
    farthest_corrs = topk_shifts(shifts, k=10)
    corr_string = get_farthest_dataset(farthest_corrs, file_list, args.target_error)
    print(shifts)
    print(corr_string)

if __name__=="__main__":
    main()
