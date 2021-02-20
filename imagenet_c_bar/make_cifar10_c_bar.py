# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torchvision as tv
from transform_finder import build_transform
from utils.converters import PilToNumpy, NumpyToPil
import os
import numpy as np
import torch

parser = argparse.ArgumentParser(description="Make CIFAR-10-C-Bar")
parser.add_argument('--cifar_dir', type=str, required=True,
        help='The path to the CIFAR-10 dataset.  This path should contain '
        'the folder cifar-10-batches-py/')
parser.add_argument('--out_dir', type=str, default='.',
        help='The path to where CIFAR-10-C will be saved.')
parser.add_argument('--num_workers', type=int, default=10,
        help='The number of workers to build images with.')
parser.add_argument('--batch_size', type=int, default=200,
        help='Batch size of torch data loader used to parallelize '
        'data processing.')
parser.add_argument('--seed', type=int, default=0,
        help='The random seed used to generate corruptions.')
parser.add_argument('--corruption_file', type=str, default='imagenet_c_bar.csv',
        help='A file that specifies which corruptions in which severities '
        'to produce.  Path is relative to the script.')

def read_corruption_csv(filename):
    with open(filename) as f:
        lines = [l.rstrip() for l in f.readlines()]
    corruptions = {}
    for line in lines:
        vals = line.split(",")
        if not vals:
            continue
        corruptions[vals[0]] = [float(v) for v in vals[1:]]
    return corruptions

def main():
    args = parser.parse_args()
    dataset_path = args.cifar_dir
    out_dir = os.path.join(args.out_dir, 'CIFAR-10-C-Bar')
    bs = args.batch_size
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    file_dir = os.path.dirname(os.path.realpath(__file__))
    corruption_csv = os.path.join(file_dir, 'cifar10_c_bar.csv')
    corruptions = read_corruption_csv(corruption_csv)

    for name, severities in corruptions.items():
        data = np.zeros((len(severities)*10000, 32, 32, 3)).astype(np.uint8)
        labels = np.zeros(len(severities)*10000).astype(np.int)
        for i, severity in enumerate(severities):
            print("Starting {}-{:.2f}...".format(name, severity))
            transform = tv.transforms.Compose([
                PilToNumpy(),
                build_transform(name=name, severity=severity, dataset_type='cifar'),
                ])
            dataset = tv.datasets.CIFAR10(dataset_path, train=False, download=False, transform=transform)
            loader = torch.utils.data.DataLoader(
                    dataset,
                    shuffle=False,
                    sampler=None,
                    drop_last=False,
                    pin_memory=False,
                    num_workers=args.num_workers,
                    batch_size=bs
                    )
            for j, (im, label) in enumerate(loader):
                if im.size(0)==bs:
                    data[i*10000+j*bs:i*10000+bs*(j+1),:,:,:] = im.numpy().astype(np.uint8)
                    labels[i*10000+j*bs:i*10000+bs*(j+1)] = label.numpy()
                else:
                    data[i*10000+j:,:,:,:] = im.numpy().astype(np.uint8)
                    labels[i*10000+j:] = label.numpy()

        out_file = os.path.join(out_dir, name + ".npy")
        print("Saving {} to {}.".format(name, out_file))
        np.save(out_file, data)
    
    labels_file = os.path.join(out_dir, "labels.npy")
    np.save(labels_file, labels)





if __name__=="__main__":
    main()
