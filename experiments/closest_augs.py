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

def individual_sort(aug_dists):
    out = []
    included = set()
    arg_sort = np.argsort(aug_dists, axis=0)
    sort = np.sort(aug_dists, axis=0)
    for row in range(len(arg_sort)):
        curr_arg = arg_sort[row]
        curr_dists = sort[row]
        sorted_args = curr_arg[np.argsort(curr_dists)]
        for i in sorted_args:
            if i not in included:
                out.append(i)
                included.add(i)
    return np.array(out)

@hydra.main(config_path="conf/closest_augs.yaml")
def train(cfg):

    np.random.seed(cfg.rng_seed)
    torch.manual_seed(cfg.rng_seed)

    log.info(cfg.pretty())
    model = instantiate(cfg.model).cuda()
    optimizer = instantiate(cfg.optim, model.parameters())
    lr_policy = instantiate(cfg.optim.lr_policy)
    if cfg.transform_file and os.path.exists(cfg.transform_file):
        log.info("Transforms found, loading feature extractor is unnecessary.  Skipping.")
    else:
        feature_extractor = instantiate(cfg.ft)
        feature_extractor.train()
    
    if cfg.transform_file and os.path.exists(cfg.transform_file):
        log.info("Transforms found, feature extraction is unnecessary.  Skipping.")
    elif cfg.aug_feature_file and os.path.exists(cfg.aug_feature_file):
        log.info("Found feature file.  Loading from {}".format(cfg.aug_feature_file))
        data = np.load(cfg.aug_feature_file)
        augmentation_features = data['features']
        indices = data['indices']
        transforms = data['transforms']
    else:
        ft_augmentation_dataset = instantiate(cfg.ft_augmentation)
        transforms = ft_augmentation_dataset.transform_list
        indices = np.random.choice(np.arange(len(ft_augmentation_dataset)), size=cfg.num_images, replace=False)
        ft_augmentation_dataset = ft_augmentation_dataset.serialize(indices)
        augmentation_features = extract_features(feature_extractor,
                                                 ft_augmentation_dataset,
                                                 cfg.ft_augmentation.batch_size,
                                                 cfg.data_loader,
                                                 average=True,
                                                 average_num=len(indices))
        if cfg.aug_feature_file:
            np.savez(cfg.aug_feature_file, 
                    features=augmentation_features, 
                    indices=indices, 
                    transforms=transforms)

    if cfg.transform_file and os.path.exists(cfg.transform_file):
        log.info("Found transform file.  Loading from {}.".format(cfg.transform_file))
        sorted_transforms = np.load(cfg.transform_file)
    else:    
        aug_strings = cfg.ft_corrupt.aug_string.split("--")
        distances = np.zeros((len(augmentation_features), len(aug_strings)))
        for i, aug in enumerate(aug_strings):
            with omegaconf.open_dict(cfg):
                ft_corrupt_dataset = instantiate(cfg.ft_corrupt, aug_string=aug)       
            if cfg.num_corrupt_images and i==0:
                indices = np.random.choice(np.arange(len(ft_corrupt_dataset)), size=cfg.num_corrupt_images, replace=False)
            ft_corrupt_dataset = ft_corrupt_dataset.serialize(indices)
            corruption_features = extract_features(feature_extractor,
                                                   ft_corrupt_dataset,
                                                   cfg.ft_corrupt.batch_size,
                                                   cfg.data_loader,
                                                   average=True)
            
            corruption_features = corruption_features.reshape(1, -1)
            dists = np.linalg.norm(augmentation_features - corruption_features, axis=-1)

            distances[:,i] = dists

        sorted_dist_args = individual_sort(distances)
        sorted_transforms = transforms[sorted_dist_args]
        if cfg.transform_file:
            np.save(cfg.transform_file, sorted_transforms)

    train_dataset = instantiate(cfg.train)
    if cfg.selection_type == 'closest':
        train_dataset.transform_list = sorted_transforms[cfg.offset:cfg.offset+cfg.num_transforms]
    elif cfg.selection_type == 'farthest':
        train_dataset.transform_list = sorted_transforms[-cfg.offset-cfg.num_transforms:-cfg.offset]\
                if cfg.offset != 0 else sorted_transforms[-cfg.num_transforms:]
    else:
        train_dataset.transform_list = sorted_transforms[np.random.choice(np.arange(len(sorted_transforms)), size=cfg.num_transforms, replace=False)]

    test_dataset = instantiate(cfg.test)

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
