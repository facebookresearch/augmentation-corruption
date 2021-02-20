# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transform_finder import build_transform
import torch
import torchvision as tv
from utils.converters import PilToNumpy, NumpyToTensor

CIFAR_MEAN = [125.3/255, 123.0/255, 113.9/255]
CIFAR_STD = [63.0/255, 62.1/255, 66.7/255]

#This is in RGB order since that is the standard for PIL
IM_MEAN = [0.485, 0.456, 0.406]
IM_STD = [0.229, 0.224, 0.225]

def read_corruption_csv(filename):
    with open(filename) as f:
        lines = [l.rstrip() for l in f.readlines()]
    corruptions = []
    for line in lines:
        vals = line.split(",")
        if not vals:
            continue
        corruptions.extend([(vals[0], float(v)) for v in vals[1:]])
    return corruptions

@torch.no_grad()
def test_c_bar(
        model,
        dataset_type,
        dataset_path,
        batch_size,
        corruption_string=None,
        loader_kwargs={},
        logger=None,
        calculate_averages=True,
        distributed=False,
        num_gpus=1
        ):

    assert dataset_type in ['imagenet', 'cifar'],\
            "Only ImageNet and CIFAR-10 are supported."
    if corruption_string is None:
        corruption_filename = 'imagenet_c_bar.csv' if dataset_type=='imagenet'\
                else 'cifar10_c_bar.csv'
        corruptions = read_corruption_csv(corruption_filename)
    else:
        corruptions = [(c.split("-")[0], float(c.split("-")[1])) for c in corruption_string.split("--")]

    results = {}
    for name, severity in corruptions:
        if dataset_type=='imagenet':
            transform = tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                PilToNumpy(),
                build_transform(name=name, severity=severity, dataset_type=dataset_type),
                NumpyToTensor(),
                tv.transforms.Normalize(IM_MEAN, IM_STD)
                ])
            path = os.path.join(dataset_path, 'val')
            dataset = tv.datasets.ImageFolder(path, transform=transform)
        elif dataset_type=='cifar':
            transform = tv.transforms.Compose([
                PilToNumpy(),
                build_transform(name=name, severity=severity, dataset_type=dataset_type),
                NumpyToTensor(),
                tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                ])
            dataset = tv.datasets.CIFAR10(dataset_path, train=False, download=False, transform=transform)

        sampler = torch.utils.data.distributed.DistributedSampler(dataset)\
                if distributed and num_gpus > 1 else None
        loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                drop_last=False,
                **loader_kwargs
                )

        num_correct = 0
        for curr_iter, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            preds = model(inputs)
            correct = torch.sum(torch.argmax(preds, dim=1)==labels)
            if distributed and num_gpus > 1:
                torch.distributed.all_reduce(correct)
            num_correct += correct.item()

        err = 100 * (1 - num_correct / len(dataset))
        corruption_string = "{}-{:.2f}".format(name, severity)
        if logger:
            logger.info("Top1 Error for {}: {:.2f}".format(corruption_string, err))
        results[corruption_string] = err

    if calculate_averages:
        import numpy as np
        unique_corruption_names = list(set([c.split("-")[0] for c in results]))
        avg_errs = {"{}-avg".format(u) : np.mean([results[c] for c in results if c.split("-")[0]==u])
                for u in unique_corruption_names}
        overall_err = np.mean(list(results.values()))
        results.update(avg_errs)
        results['overall-avg'] = overall_err
        if logger:
            for k,v in avg_errs.items():
                logger.info("Top1 Error for {}: {:.2f}".format(k,v))
            logger.info("Average Top1 Error: {}".format(overall_err))
    return results

