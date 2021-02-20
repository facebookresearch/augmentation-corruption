# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
from .utils import logging as lu
import numpy as np
import os

log = logging.getLogger(__name__)

def distributed_gather_features(curr_features, batch_size, num_gpus):
    gather_list = [torch.zeros((batch_size, curr_features.size(-1)), device=curr_features.device)\
            for i in range(num_gpus)]
    count = curr_features.size(0)
    if count < batch_size:
        curr_features = torch.cat((curr_features, torch.zeros((batch_size - count, curr_features.size(-1)), device=curr_features.device)), dim=0)

    torch.distributed.all_gather(gather_list, curr_features)
    count = torch.Tensor([count]).cuda()
    torch.distributed.all_reduce(count)
    count = int(count.item())

    # Here we use that the distributed data sampler interleaves sampling across replicas
    curr_features = torch.stack(gather_list, dim=1).reshape(-1, curr_features.size(-1))
    curr_features = curr_features[:count,:]
    return curr_features


def extract_features(feature_extractor, dataset, batch_size, loader_params, average=True, num_gpus=1, average_num=None, preemption_protection=False, is_leader=True):

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)\
            if num_gpus > 1 else None
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=loader_params.num_workers,
            pin_memory=loader_params.pin_memory,
            drop_last=False
            )

    features = None
    count = 0
    starting_iter = -1
    if preemption_protection and os.path.exists('feature_extraction.tmp.npz'):
        data = np.loadz('feature_extraction.tmp.npz')
        features = torch.Tensor(data['features']).cuda()
        count = data['count']
        starting_iter = data['curr_iter']
    for curr_iter, (inputs, labels) in enumerate(loader):
        if preemption_protection and curr_iter <= starting_iter:
            continue
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        curr_features = feature_extractor.extract(inputs)
        if average and average_num is None:
            curr_features = torch.sum(curr_features, dim=0)
            if num_gpus > 1:
                torch.distributed.all_reduce(curr_features)
            features = (features + curr_features.detach().cpu()) if features is not None else curr_features.detach().cpu()
        elif average:
            num_features = len(dataset) // average_num
            if num_gpus > 1:
                curr_features = distributed_gather_features(curr_features, batch_size, num_gpus)
            if features is None:
                features = torch.zeros(num_features, curr_features.size(-1))
            if count + curr_features.size(0) > num_features:
                remainder = count + curr_features.size(0) - num_features
                features[count:, :] += curr_features[:num_features-count,:].detach().cpu()
                offset = 0
                while remainder > num_features:
                    features += curr_features[offset+num_features-count:2*num_features-count+offset].detach().cpu()
                    offset += num_features
                    remainder -= num_features
                features[:remainder,:] += curr_features[offset+num_features-count:,:].detach().cpu()
                count = remainder
            else:
                features[count:count+curr_features.size(0),:] += curr_features.detach().cpu()
                count += curr_features.size(0)
                count = count % num_features

        else:
            if num_gpus > 1:
                curr_features = distributed_gather_features(curr_features, batch_size, num_gpus)
                
            if features is None:
                features = torch.zeros(len(dataset), curr_features.size(-1))
            features[count:count+curr_features.size(0),:] = curr_features.detach().cpu()
            count += curr_features.size(0)

        if preemption_protection and curr_iter % 5000 == 0 and is_leader:
            np.savez('feature_extraction.tmp.npz', features=features.detach().cpu().numpy(), count=count, curr_iter=curr_iter)
    if average and average_num is None:
        features /= len(dataset)
    elif average:
        features /= average_num

    return features.detach().cpu().numpy()

