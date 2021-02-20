# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
from  .utils import logging as lu
from omegaconf import open_dict
from .augmentations.utils import aug_finder
from hydra.utils import instantiate
import numpy as np
import os
import pickle


log = logging.getLogger(__name__)

def test_corrupt_net(model, corrupt_cfg, batch_size, loader_params, aug_string=None, mCE_denom=None, clean_err=None, imagenetc_grouping=True, num_gpus=1, log_name=None):

    model.eval()
    if aug_string is None:
        augs = aug_finder.get_augs_by_tag(['imagenet_c'])
        severities = [1,2,3,4,5]
        augs = ["{}-{}".format(a.name, s) for a in augs for s in severities]
    else:
        augs = aug_string.split("--")


    if log_name is not None and os.path.exists(log_name):
        prestats = lu.load_json_stats(log_name)
    else:
        prestats = None
        

    errs = []
    for aug in augs:
        if prestats is not None and len(lu.parse_json_stats(prestats, row_type=aug, key='top1_err')) > 0:
            continue
        with open_dict(corrupt_cfg):
            corrupt_dataset = instantiate(corrupt_cfg, aug_string=aug)
        sampler = torch.utils.data.distributed.DistributedSampler(corrupt_dataset)\
                if num_gpus > 1 else None
        loader = torch.utils.data.DataLoader(
                corrupt_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=loader_params.num_workers,
                pin_memory=loader_params.pin_memory,
                drop_last=False
                )
        num_correct = 0
        for curr_iter, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            preds = model(inputs)
            correct = torch.sum(torch.argmax(preds, dim=1)==labels)
            if num_gpus > 1:
                torch.distributed.all_reduce(correct)
            num_correct += correct.item()

        err = 100 * (1 - num_correct / len(corrupt_dataset))
        stats = {'_type' : aug, 'top1_err' : err}
        lu.log_json_stats(stats)
        errs.append(err)


    # Calculating records
    if mCE_denom is not None:
        mCE_denom = pickle.load(open(os.path.join(os.path.dirname(__file__), '../baseline_data/', mCE_denom), 'rb'))

    errs = np.array(errs)
    aug_names = [a.split("-")[0] for a in augs]
    unique_aug_names = list(set(aug_names))
    avg_errs = [np.mean(errs[[i for i, a in enumerate(aug_names) if a==u]]) for u in unique_aug_names]
    avg_errs = np.array(avg_errs)
    mCE = None
    rmCE = None
    if mCE_denom:
        mCE = [100 * avg_errs[i] / mCE_denom[a] for i, a in enumerate(unique_aug_names)]
        mCE = np.array(mCE)
        if clean_err:
            rmCE = [100 * (avg_errs[i] - clean_err) / (mCE_denom[a] - mCE_denom['clean'])\
                    for i, a in enumerate(unique_aug_names)]
            rmCE = np.array(rmCE)
    for i, a in enumerate(unique_aug_names):
        stats = {'_type' : a + '-avg', 'top1_err' : avg_errs[i]}
        if mCE is not None:
            stats['mCE'] = mCE[i]
        if rmCE is not None:
            stats['rmCE'] = rmCE[i]
        lu.log_json_stats(stats)
    if imagenetc_grouping:
        for aug_type in ['blur', 'digital', 'noise', 'weather', 'extra']:
            aug_indices = [i for i, a in enumerate(unique_aug_names)\
                    if aug_type in aug_finder.get_aug_by_name(a).tags]
            err_for_type = np.mean(avg_errs[aug_indices])
            stats = {'_type' : aug_type + '-avg', 'top1_err' : err_for_type}
            if mCE is not None:
                mCE_for_type = np.mean(mCE[aug_indices])
                stats['mCE'] = mCE_for_type
            if rmCE is not None:
                rmCE_for_type = np.mean(rmCE[aug_indices])
                stats['rmCE'] = rmCE_for_type
            lu.log_json_stats(stats)

    if imagenetc_grouping:
        indices = [i for i, a in enumerate(unique_aug_names)\
                if 'extra' not in aug_finder.get_aug_by_name(a).tags]
    else:
        indices = [i for i, a in enumerate(unique_aug_names)]
        
    overall_avg = np.mean(avg_errs[indices])
    stats = {'_type' : 'overall-avg', 'top1_err' : overall_avg}
    if mCE is not None:
        overall_mCE = np.mean(mCE[indices])
        stats['mCE'] = overall_mCE
    if rmCE is not None:
        overall_rmCE = np.mean(rmCE[indices])
        stats['rmCE'] = overall_rmCE
    lu.log_json_stats(stats)
