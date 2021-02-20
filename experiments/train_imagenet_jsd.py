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
import submitit

log = logging.getLogger(__name__)

@hydra.main(config_path="conf/train_imagenet_jsd.yaml")
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


def train(cfg, is_leader):

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
    if cfg.optim.max_epoch > 0 and cfg.train.weights is None:
        print("Loading training set...")
        train_dataset = instantiate(cfg.train)
    else:
        print("Skipping loading the training dataset, 0 epochs of training to perform "
        " or pre-trained weights provided.")
        train_dataset = None
    print("Loading test set...")
    test_dataset = instantiate(cfg.test)
    lr_policy = instantiate(cfg.optim.lr_policy)  

    print("Training...")
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
            is_leader=is_leader,
            jsd_num=cfg.train.params.jsd_num,
            jsd_alpha=cfg.train.jsd_alpha
            )

    print("Testing...")
    err = test_net(model=model,
            test_dataset=test_dataset,
            batch_size=cfg.test.batch_size,
            loader_params=cfg.data_loader,
            num_gpus=cfg.num_gpus)

    test_corrupt_net(model=model,
            corrupt_cfg=cfg.corrupt,
            batch_size=cfg.corrupt.batch_size,
            loader_params=cfg.data_loader,
            aug_string=cfg.corrupt.aug_string,
            clean_err=err,
            mCE_denom=cfg.corrupt.mCE_baseline_file,
            num_gpus=cfg.num_gpus,
            log_name='train_imagenet.log')





if __name__=="__main__":
    run()
