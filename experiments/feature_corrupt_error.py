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
import os
import omegaconf
from overlap.extract_features import extract_features
from overlap.utils import logging as lu

log = logging.getLogger(__name__)

@hydra.main(config_path="conf/feature_corrupt_error.yaml")
def train(cfg):

    np.random.seed(cfg.rng_seed)
    torch.manual_seed(cfg.rng_seed)

    log.info(cfg.pretty())
    model = instantiate(cfg.model).cuda()
    optimizer = instantiate(cfg.optim, model.parameters())
    train_dataset = instantiate(cfg.train)
    test_dataset = instantiate(cfg.test)
    lr_policy = instantiate(cfg.optim.lr_policy)
    feature_extractor = instantiate(cfg.ft)
    feature_extractor.train()
    
    if cfg.aug_feature_file and os.path.exists(cfg.aug_feature_file):
        log.info("Found feature file.  Loading from {}".format(cfg.aug_feature_file))
        data = np.load(cfg.aug_feature_file)
        augmentation_features = data['features']
        indices = data['indices']
    else:
        ft_augmentation_dataset = instantiate(cfg.ft_augmentation)
        indices = np.random.choice(np.arange(len(ft_augmentation_dataset)), size=cfg.num_images, replace=False)
        ft_augmentation_dataset = ft_augmentation_dataset.serialize(indices)
        augmentation_features = extract_features(feature_extractor,
                                                 ft_augmentation_dataset,
                                                 cfg.ft_augmentation.batch_size,
                                                 cfg.data_loader,
                                                 average=True,
                                                 average_num=len(indices))
        #nf, lf = augmentation_features.shape
        #augmentation_features = np.mean(augmentation_features.reshape(len(indices), nf//len(indices), lf), axis=0)
        if cfg.aug_feature_file:
            np.savez(cfg.aug_feature_file, features=augmentation_features, indices=indices)


    aug_strings = cfg.ft_corrupt.aug_string.split("--")
    for aug in aug_strings:
        with omegaconf.open_dict(cfg):
            ft_corrupt_dataset = instantiate(cfg.ft_corrupt, aug_string=aug)
        ft_corrupt_dataset = ft_corrupt_dataset.serialize(indices)
        corruption_features = extract_features(feature_extractor,
                                               ft_corrupt_dataset,
                                               cfg.ft_corrupt.batch_size,
                                               cfg.data_loader,
                                               average=True,
                                               average_num=len(indices))
        nf, lf = corruption_features.shape
        #corruption_features = np.mean(corruption_features.reshape(len(indices), nf//len(indices), lf), axis=0)
        
        augmentation_features = augmentation_features.reshape(-1, 1, lf)
        corruption_features = corruption_features.reshape(1, -1, lf)
        mean_aug = np.mean(augmentation_features.reshape(-1,lf), axis=0)
        mean_corr = np.mean(corruption_features.reshape(-1,lf), axis=0)
        mmd = np.linalg.norm(mean_aug-mean_corr, axis=0)
        msd = np.min(np.linalg.norm(augmentation_features.reshape(-1,lf)-mean_corr.reshape(1,lf),axis=1),axis=0)

        stats = {"_type" : aug,
                "mmd" : str(mmd),
                "msd" : str(msd),
                }
        lu.log_json_stats(stats)


    train_net(model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            batch_size=cfg.train.batch_size,
            max_epoch=cfg.optim.max_epoch,
            loader_params=cfg.data_loader,
            lr_policy=lr_policy,
            save_period=cfg.train.checkpoint_period,
            weights=cfg.train.weights
            )

    err = test_net(model=model,
            test_dataset=test_dataset,
            batch_size=cfg.test.batch_size,
            loader_params=cfg.data_loader,
            output_name='test_epoch')

    test_corrupt_net(model=model,
            corrupt_cfg=cfg.corrupt,
            batch_size=cfg.corrupt.batch_size,
            loader_params=cfg.data_loader,
            aug_string=cfg.corrupt.aug_string,
            clean_err=err,
            mCE_denom=cfg.corrupt.mCE_baseline_file)

if __name__=="__main__":
    train()
