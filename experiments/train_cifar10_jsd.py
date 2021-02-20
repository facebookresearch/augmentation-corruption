# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from hydra.utils import instantiate
import logging
from overlap.train_net_jsd import train_net
from overlap.test_net import test_net
from overlap.test_corrupt_net import test_corrupt_net
import numpy as np
import torch
import pickle
from pathlib import Path

log = logging.getLogger(__name__)

@hydra.main(config_path="conf/train_cifar10_jsd.yaml")
def train(cfg):

    np.random.seed(cfg.rng_seed)
    torch.manual_seed(cfg.rng_seed)

    log.info(cfg.pretty())
    model = instantiate(cfg.model).cuda()
    optimizer = instantiate(cfg.optim, model.parameters())
    train_dataset = instantiate(cfg.train)
    test_dataset = instantiate(cfg.test)
    lr_policy = instantiate(cfg.optim.lr_policy)  

    train_net(model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            batch_size=cfg.train.batch_size,
            max_epoch=cfg.optim.max_epoch,
            loader_params=cfg.data_loader,
            lr_policy=lr_policy,
            save_period=cfg.train.checkpoint_period,
            weights=cfg.train.weights,
            jsd_num=cfg.train.params.jsd_num,
            jsd_alpha=cfg.train.jsd_alpha
            )

    err = test_net(model=model,
            test_dataset=test_dataset,
            batch_size=cfg.test.batch_size,
            loader_params=cfg.data_loader)


    test_corrupt_net(model=model,
            corrupt_cfg=cfg.corrupt,
            batch_size=cfg.corrupt.batch_size,
            loader_params=cfg.data_loader,
            aug_string=cfg.corrupt.aug_string,
            clean_err=err,
            mCE_denom=cfg.corrupt.mCE_baseline_file)





if __name__=="__main__":
    train()
