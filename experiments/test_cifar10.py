# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from hydra.utils import instantiate
import logging
from overlap.train_net import train_net
from overlap.test_net import test_net
from overlap.test_corrupt_net import test_corrupt_net
import numpy as np
import torch
import pickle

log = logging.getLogger(__name__)

@hydra.main(config_path="conf/test_cifar10.yaml")
def train(cfg):

    np.random.seed(cfg.rng_seed)
    torch.manual_seed(cfg.rng_seed)

    log.info(cfg.pretty())
    model = instantiate(cfg.model).cuda()
    test_dataset = instantiate(cfg.test)

    checkpoint = torch.load(cfg.weights, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])

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
