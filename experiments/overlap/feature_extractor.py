# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from hydra.utils import instantiate
from .train_net import train_net

class Network(object):
    
    def __init__(self, model_cfg, optim_cfg, dataset_cfg, data_loader, num_gpus=1, is_leader=True):
        cur_device = torch.cuda.current_device()
        self.model = instantiate(model_cfg).cuda(device=cur_device)
        if num_gpus > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                    module=self.model,
                    device_ids=[cur_device],
                    output_device=cur_device
                    )
        self.optimizer = instantiate(optim_cfg, self.model.parameters())
        if optim_cfg.max_epoch > 0:
            self.dataset = instantiate(dataset_cfg)
        else:
            self.dataset = None
        self.batch_size = dataset_cfg.batch_size
        self.max_epoch = optim_cfg.max_epoch
        self.loader_params = data_loader
        self.lr_policy = instantiate(optim_cfg.lr_policy)
        self.save_period = dataset_cfg.checkpoint_period
        self.weights = dataset_cfg.weights
        self.num_gpus = num_gpus
        self.is_leader = is_leader

    def train(self):
        train_net(self.model,
                self.optimizer,
                self.dataset,
                self.batch_size,
                self.max_epoch,
                self.loader_params,
                self.lr_policy,
                save_period=self.save_period,
                name='ft',
                weights=self.weights,
                num_gpus=self.num_gpus,
                is_leader=self.is_leader
                )
        self.model.eval()

        

    def extract(self, x):
        preds = self.model(x)
        if self.num_gpus > 1:
            return self.model.module.features
        else:
            return self.model.features

