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
from PIL import Image

parser = argparse.ArgumentParser(description="Make CIFAR-10-C-Bar")
parser.add_argument('--imagenet_dir', type=str, required=True,
        help='The path to the ImageNet dataset.  This path should contain '
        'the folder val/')
parser.add_argument('--out_dir', type=str, default='.',
        help='The path to where ImageNet-C will be saved.')
parser.add_argument('--num_workers', type=int, default=10,
        help='The number of workers to build images with.')
parser.add_argument('--batch_size', type=int, default=100,
        help='Batch size of torch data loader used to parallelize '
        'data processing.')
parser.add_argument('--corruption_file', type=str, default='imagenet_c_bar.csv',
        help='A file that specifies which corruptions in which severities '
        'to produce. Path is relative to the script.')
parser.add_argument('--seed', type=int, default=0,
        help='The random seed used to generate corruptions.')

class SavingDataset(tv.datasets.ImageFolder):

    def __init__(self, root, out_dir, transform=None):
        super(SavingDataset, self).__init__(root, transform=transform)
        self.out_dir = out_dir

    def __getitem__(self, index):
        image, label = super(SavingDataset, self).__getitem__(index)
        class_name = self.classes[label]
        out_dir = os.path.join(self.out_dir, class_name)
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            pass
        file_name = os.path.basename(self.samples[index][0])
        save_path = os.path.join(out_dir, file_name)
        Image.fromarray(np.uint8(image)).save(save_path, quality=85, optimize=True)
        return image, label



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
    dataset_path = args.imagenet_dir
    corruption_file = args.corruption_file
    out_dir = os.path.join(args.out_dir, 'ImageNet-C-Bar')
    np.random.seed(args.seed)
    bs = args.batch_size
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    file_dir = os.path.dirname(os.path.realpath(__file__))
    corruption_csv = os.path.join(file_dir, corruption_file)
    corruptions = read_corruption_csv(corruption_csv)

    for name, severities in corruptions.items():
        corruption_dir = os.path.join(out_dir, name)
        if not os.path.exists(corruption_dir):
            os.mkdir(corruption_dir)
        for i, severity in enumerate(severities):
            severity_dir = os.path.join(corruption_dir, "{:.2f}".format(severity))
            if not os.path.exists(severity_dir):
                os.mkdir(severity_dir)
            print("Starting {}-{:.2f}...".format(name, severity))
            transform = tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                PilToNumpy(),
                build_transform(name=name, severity=severity, dataset_type='imagenet'),
                ])
            path = os.path.join(dataset_path, 'val')
            dataset = SavingDataset(path, severity_dir, transform=transform)
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
                if (j+1) % 10 == 0:
                    print("Completed {}/{}".format(j, len(loader)))





if __name__=="__main__":
    main()
