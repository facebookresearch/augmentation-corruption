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

parser = argparse.ArgumentParser(description="Collect run summaries.")
parser.add_argument('--new_corr_dir', dest='data_dir', required=True)
parser.add_argument('--baseline_corr_dir', dest='baseline_dir', required=True)
parser.add_argument('--precision', type=float, dest='precision', default=0.01)
parser.add_argument('--target_error', type=float, dest='target', required=True)
parser.add_argument('--num', type=int, dest='num', default=5)
parser.add_argument('--out', dest='out', required=True)
parser.add_argument('--log_name', default='severity_scan.log', dest='log_name')

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


def get_average_spread(baseline_errs):
    '''
    Calculate the average spread in severity in the baseline data, so
    the new corruption datasets can attempt to match it.
    '''
    bcorrs = sorted(list(set([c.split("-")[0] for c in baseline_errs.keys()])))
    avg = 0.0
    for bcorr in bcorrs:
        lower = abs((baseline_errs["{}-1".format(bcorr)] / baseline_errs["{}-3".format(bcorr)] - 1))
        upper = abs((baseline_errs["{}-5".format(bcorr)] / baseline_errs["{}-3".format(bcorr)] - 1))
        avg += (lower + upper) / 2
    return avg / len(bcorrs)

def build_sets(corr_errs, avg_spread):
    '''
    For each severity 3-8, associate a set of 5 severities with it that
    best match the average spread, where that severity is the middle of
    the five.

    Inputs:
    corr_errs: dictionary where each key is a string "{corr}-{severity}"
      and each value is the test error.
    avg_spread: float specifying the average spread to try to match

    Output:
    dictionary where each key is a string giving the corruption name, 
      and each value is a list of 5-tuples giving all sets of 5 severities
      associated to that corruption.
    '''
    corrs = sorted(list(set([c.split("-")[0] for c in corr_errs.keys()])))
    corr_sets = {c : [] for c in corrs}
    for c in corrs:
        sevs = sorted([float(i.split("-")[1]) for i in corr_errs.keys() if c == i.split("-")[0]])
        for i in np.arange(2, len(sevs)-2):
            # Sev 1
            best = float('inf')
            best_match_s1 = None
            for j in np.arange(0, i-1):
                sep = corr_errs["{}-{}".format(c, sevs[j])] / corr_errs["{}-{}".format(c, sevs[i])] - 1
                sep_sep = abs(-avg_spread - sep)
                if sep_sep <= best:
                    best = sep_sep
                    best_match_s1 = j
            # Sev 2
            best = float('inf')
            best_match_s2 = None
            for j in np.arange(best_match_s1+1, i):
                sep = corr_errs["{}-{}".format(c, sevs[j])] / corr_errs["{}-{}".format(c, sevs[i])] - 1
                sep_sep = abs(-avg_spread/2 - sep)
                if sep_sep <= best:
                    best = sep_sep
                    best_match_s2 = j

            # Sev 5
            best = float('inf')
            best_match_s5 = None
            for j in np.arange(i+2, len(sevs)):
                sep = corr_errs["{}-{}".format(c, sevs[j])] / corr_errs["{}-{}".format(c, sevs[i])] - 1
                sep_sep = abs(avg_spread - sep)
                if sep_sep <= best:
                    best = sep_sep
                    best_match_s5 = j

            # Sev 4
            best = float('inf')
            best_match_s4 = None
            for j in np.arange(i+1, best_match_s5):
                sep = corr_errs["{}-{}".format(c, sevs[j])] / corr_errs["{}-{}".format(c, sevs[i])] - 1
                sep_sep = abs(avg_spread/2 - sep)
                if sep_sep <= best:
                    best = sep_sep
                    best_match_s4 = j

            corr_sets[c].append((sevs[best_match_s1], sevs[best_match_s2], sevs[i], sevs[best_match_s4], sevs[best_match_s5]))
    return corr_sets


def build_distance_table(baseline_features, corr_features):
    '''
    Calculates distances between corruption features.  Across baseline
    features and severities, takes the minimum distance, across
    severities in the new corruption set, takes the average.

    Inputs:
    baseline_features: 3d numpy array ordered as
      [corruption, severity, features]
    corr_features: 4d numpy array ordered as
      [corruption, severity_set, severity, features]

    Output
    2d numpy array ordered as [corruption, severity_set]
    '''
    nc, nss, ns, nf = corr_features.shape
    corr_features = corr_features.reshape(nc, nss, ns, 1, 1, nf)
    nb, _, _ = baseline_features.shape
    baseline_features = baseline_features.reshape(1, 1, 1, nb, ns, nf)
    dists = np.linalg.norm(corr_features - baseline_features, axis=-1)
    min_dists = np.mean(np.min(np.min(dists, axis=-1), axis=-1), axis=-1)

    return min_dists

def build_corr_tables(corr_sets, corr_errs, corr_features):
    '''
    Given a list of corruption severity sets, builds the tables that
    will be used to calculate distance.  For each corruption, the tables
    are ordered in increasing order, since this is required to make the
    dataset selection algorithm more efficient.  This ordering is also
    returned so it can be reversed at the end.

    Inputs:
    corr_sets: dictionary of corruption keys with lists of severity set
      values
    corr_errs: dictionary of keys with the form '{corr}-{severity}' and
      values that are the errors on that corruption/severity pair
    corr_features: dictionary of keys with the form '{corr}-{severity}'
      and values that are the features on that corruption/severity pair

    Outputs:
    1. A list of all corruption strings, in the order they appear the
       table.
    2. dictionary where the keys are corruption strings, and the values
       are the severity sets ordered by increasing corruption error.
    3. 2d numpy array with the shape [corruption, severity_set] that
       gives the average error on that severity set and corruption
    4. 4d numpy array with the shape 
       [corruption, severity_set, severity, features]
    '''

    corrs = list(corr_sets.keys())
    ordered = {}
    len_feats = len(list(corr_features.values())[0])
    err_table = np.zeros((len(corrs), len(corr_sets[corrs[0]])))
    feat_table = np.zeros((len(corrs), len(corr_sets[corrs[0]]), len(corr_sets[corrs[0]][0]), len_feats))
    for i, corr in enumerate(corrs):
        curr_errs = np.zeros(len(corr_sets[corr]))
        curr_feats = np.zeros((len(corr_sets[corr]), len(corr_sets[corrs[0]][0]), len_feats))
        for j, sev_list in enumerate(corr_sets[corr]):
            curr_errs[j] = np.mean([corr_errs["{}-{}".format(corr, s)] for s in sev_list])
            curr_feats[j,:,:] = np.array([corr_features["{}-{}".format(corr, s)] for s in sev_list])
        sev_order = np.argsort(curr_errs)
        err_table[i,:] = np.sort(curr_errs)
        feat_table[i, :, :, :] = curr_feats[sev_order, :, :]
        ordered[corr] = np.array(corr_sets[corr])[sev_order]
    return corrs, ordered, err_table, feat_table

def build_baseline_features(baseline_features):
    '''
    Builds a table of baseline corruption features, given a dictionary.

    Inputs:
    baseline_features: dictionary of features with keys that are strings
      as "{corr}-{severity}"

    Outputs:
    3d numpy array ordered as [corruption, severity, features].

    '''
    corrs = sorted(list(set([v.split("-")[0] for v in baseline_features.keys()])))
    sevs = sorted(list(set([int(v.split("-")[1]) for v in baseline_features.keys()])))
    len_feats = len(list(baseline_features.values())[0])
    baseline_table = np.zeros((len(corrs), len(sevs), len_feats))
    for i, c in enumerate(corrs):
        for j, s in enumerate(sevs):
            baseline_table[i,j,:] = baseline_features["{}-{}".format(c,s)]
    return baseline_table
    
def sample_matched_corruptions(err_table, baseline_err, precision, num):
    '''
    Iterates over all 'num'-sized combinations of corruptions and selects
    a set of severities that has error within 'precision' of the baseline
    error.  If multiple sets of severities fall within the precision
    window, it picks one at random.  If none do, it skips that combination
    of corruptions.

    The runtime is O((num_corruptions * num_severity_sets)^num), though
    in practice the algorithm below is usually
    O(num_corruptions^num * num_severity_sets^(num-1)).

    Inputs:
    err_table: 2d numpy array of shape [corruptions, severity_sets] 
      listing the average error for each set.
    baseline_err: float giving the target error to match
    precision: float giving the percentage variation from the baseline
      error allowed for an accepted severity set
    num: int listing the number of corruptions to combine

    Output:
    A list of sampled datasets, where each sampled dataset is a list
      of 'num' 2-tuples (corruption_index, severity_set_index).
    '''
    count = 0
    total = comb(err_table.shape[0], num, exact=True)
    chosen_augs = []
    for idxs in combinations(range(err_table.shape[0]), num):
        all_augs = []
        count += 1
        if count % 1000 == 0:
            print("On iteration {}/{}".format(count, total))
        # Loop over severities for all chosen corruptions except for the
        # last two.  Since the severity sets are ordered by average error,
        # we can work from the outside in to typically save one factor of
        # 'num' in calculation time.
        for sev_idxs in product(*[range(err_table.shape[1]) for i in range(num-2)]):
            target = baseline_err * num
            err_sum = 0.0
            for i in range(num-2):
                err_sum += err_table[idxs[i], sev_idxs[i]]
            stack = [(0, err_table.shape[1]-1)] # Start on the two ends
            seen = set()
            while stack:
                i, j = stack.pop()
                if (i,j) in seen or i >= err_table.shape[1] or j < 0:
                    continue
                seen.add((i,j))
                final_err_sum = err_sum + err_table[idxs[-2],i] + err_table[idxs[-1],j]
                if abs((target-final_err_sum)/target) < precision:

                    curr = [(idxs[k], sev_idxs[k]) for k in range(num-2)] + [(idxs[-2],i), (idxs[-1],j)]
                    all_augs.append(curr)

                    stack.append([i+1, j])
                    stack.append([i, j-1])
                elif (target-final_err_sum)/target >= precision:
                    stack.append([i+1, j])
                else:
                    stack.append([i, j-1])
        if all_augs:
            idx_choice = np.random.randint(low=0, high=len(all_augs))
            chosen_augs.append(all_augs[idx_choice])
    return chosen_augs


def main():
    args = parser.parse_args()
    data_dir = args.data_dir
    baseline_dir = args.baseline_dir
    precision = args.precision
    num_corr = args.num
    out_file = args.out
    log_name = args.log_name
    target_error = args.target
    baseline_exclusions = ['saturate', 'spatter', 'gaussian_blur', 'speckle_noise']
    corr_exclusions = []

    print("Loading data...")
    data_dirs = data_dir.split(",")
    baseline_dirs = baseline_dir.split(",")
    corr_errs, corr_features = get_data(data_dirs, corr_exclusions, log_file=log_name)
    baseline_errs, baseline_features = get_data(baseline_dirs, exclusions=baseline_exclusions, log_file=log_name)

    baseline_table = build_baseline_features(baseline_features)
    avg_spread = get_average_spread(baseline_errs)



    corr_sets = build_sets(corr_errs, avg_spread)
    corrs, ordered_sev_list, err_table, feat_table = build_corr_tables(corr_sets, corr_errs, corr_features)
    dists = build_distance_table(baseline_table, feat_table)
    
    chosen = sample_matched_corruptions(err_table, target_error, precision, num_corr)
    out = []
    for aug_list in chosen:
        sub_aug_strings = []
        err = 0.0
        curr_dists = None
        for a in aug_list:
            corr = corrs[a[0]]
            sevs = ordered_sev_list[corr][a[1]]
            sub_aug_strings.append("--".join(["{}-{}".format(corr,s) for s in sevs]))
            err += err_table[a[0], a[1]]
            curr_curr_dists = dists[a[0], a[1]]
            curr_dists = np.concatenate((curr_dists, curr_curr_dists.reshape(1,-1)), axis=0) if curr_dists is not None else curr_curr_dists.reshape(1,-1)
        err /= len(aug_list)
        avg_dists = np.mean(curr_dists, axis=0)
        aug_string = "--".join(sub_aug_strings)
        data_out = ",".join([aug_string, str(err)] + [str(x) for x in avg_dists])
        out.append(data_out)

    with open(out_file, 'w') as f:
        f.write(",,".join([data_dir, baseline_dir, str(precision), str(num_corr)]))
        f.write("\n")
        f.write("\n".join(out))

if __name__=="__main__":
    main()
