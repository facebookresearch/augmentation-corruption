# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
from .utils import logging as lu

log = logging.getLogger(__name__)

def test_net(model, test_dataset, batch_size, loader_params, output_name='test_epoch', num_gpus=1):

    model.eval()
    sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)\
            if num_gpus > 1 else None
    loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=loader_params.num_workers,
            pin_memory=loader_params.pin_memory,
            drop_last=False
            )

    num_correct = 0
    num_total = 0
    for curr_iter, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = model(inputs)
        correct = torch.sum(torch.argmax(preds, dim=1)==labels)
        if num_gpus > 1:
            torch.distributed.all_reduce(correct)
        num_correct += correct.item()

    err = 100 * (1 - num_correct / len(test_dataset))
    stats = {'_type' : output_name, 'top1_err' : err}
    lu.log_json_stats(stats)

    return err
