# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from hydra.utils import instantiate
import logging
from overlap.train_net import train_net
from overlap.test_net import test_net
import numpy as np
import torch
import pickle
import os
import omegaconf
from overlap.extract_features import extract_features
import submitit

log = logging.getLogger(__name__)

@hydra.main(config_path="conf/severity_scan_imagenet.yaml")
def run(cfg):
    if cfg.num_gpus > 1:
        job_env = submitit.JobEnvironment()
        rank = job_env.global_rank
        world_size = job_env.num_tasks
        if rank != 0:
            logging.root.handlers = []
        try:
            torch.cuda.set_device(rank)
            torch.distributed.init_process_group(
                    backend='nccl',
                    init_method="tcp://{}:{}".format('localhost', 10001),
                    world_size=world_size,
                    rank=rank
                    )
            train(cfg, is_leader=(rank==0))
        except KeyboardInterrupt:
            pass
        finally:
            torch.distributed.destroy_process_group()
    else:
        train(cfg, is_leader=True)

def train(cfg, is_leader=True):

    np.random.seed(cfg.rng_seed)
    torch.manual_seed(cfg.rng_seed)

    log.info(cfg.pretty())
    cur_device = torch.cuda.current_device()
    model = instantiate(cfg.model).cuda(device=cur_device)
    if cfg.num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[cur_device],
                output_device=cur_device
                )
    optimizer = instantiate(cfg.optim, model.parameters())
    if cfg.optim.max_epoch > 0:
        train_dataset = instantiate(cfg.train)
    else:
        train_dataset = None
    test_dataset = instantiate(cfg.test)
    lr_policy = instantiate(cfg.optim.lr_policy)
    with omegaconf.open_dict(cfg):
        feature_extractor = instantiate(cfg.ft, num_gpus=cfg.num_gpus, is_leader=is_leader)
    feature_extractor.train()
    
    train_net(model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            batch_size=cfg.train.batch_size,
            max_epoch=cfg.optim.max_epoch,
            loader_params=cfg.data_loader,
            lr_policy=lr_policy,
            save_period=cfg.train.checkpoint_period,
            weights=cfg.train.weights,
            num_gpus=cfg.num_gpus,
            is_leader=is_leader
            )

    err = test_net(model=model,
            test_dataset=test_dataset,
            batch_size=cfg.test.batch_size,
            loader_params=cfg.data_loader,
            output_name='test_epoch',
            num_gpus=cfg.num_gpus)

    if os.path.exists(cfg.feature_file):
        feature_dict = {k : v for k, v in np.load(cfg.feature_file).items()}
    else:
        feature_dict = {}
    indices = np.load(cfg.ft_corrupt.indices_file)
    for aug in cfg.aug_string.split("--"):
        if len(aug.split("-")) > 1:
            #log.info("Severity provided in corrupt.aug_string will be weighted by given severity.")
            sev = aug.split("-")[1]
            if len(sev.split("_")) > 1:
                low = float(sev.split("_")[0])
                high = float(sev.split("_")[1])
            else:
                low = 0.0
                high = float(sev)

            sev_factor = (high - low) * cfg.severity / 10 + low
        else:
            sev_factor = cfg.severity
        aug = aug.split("-")[0]
        aug_string = "{}-{}".format(aug, sev_factor)
        if aug_string in feature_dict:
            continue
        with omegaconf.open_dict(cfg.corrupt):
            corrupt_dataset = instantiate(cfg.corrupt, aug_string=aug_string)
        err = test_net(model=model,
                test_dataset=corrupt_dataset,
                batch_size=cfg.corrupt.batch_size,
                loader_params=cfg.data_loader,
                output_name=aug_string,
                num_gpus=cfg.num_gpus)
        with omegaconf.open_dict(cfg.ft_corrupt):
            ft_corrupt_dataset = instantiate(cfg.ft_corrupt, aug_string=aug_string)
        if cfg.ft_corrupt.params.num_transforms is not None:
            ft_corrupt_dataset = ft_corrupt_dataset.serialize(indices)
        else:
            ft_corrupt_dataset = torch.utils.data.Subset(ft_corrupt_dataset, indices)
        
        feature = extract_features(feature_extractor=feature_extractor,
                dataset=ft_corrupt_dataset,
                batch_size=cfg.ft_corrupt.batch_size,
                loader_params=cfg.data_loader,
                average=True,
                num_gpus=cfg.num_gpus)
        feature_dict[aug_string] = feature
        if is_leader:
            np.savez(cfg.feature_file, **feature_dict)

if __name__=="__main__":
    run()
