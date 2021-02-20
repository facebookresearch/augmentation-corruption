# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Generate random indicies '\
        'for sampling from the CIFAR-10 or ImageNet training sets.")
parser.add_argument('--dataset', type=str, required=True,
        help='Should be in [\'cifar-10\', \'imagenet\'].')
parser.add_argument('--num', type=int, required=True,
        help='Number of indices to generate.')
parser.add_argument('--out', type=str, required=True,
        help='Output file.  Should be .npy format.')

def main():
    args = parser.parse_args()
    assert args.dataset in ['cifar-10', 'imagenet'], "Unknown dataset."
    max_index = 50000 if args.dataset=='cifar-10' else 1281167
    indices = np.random.choice(np.arange(max_index), size=args.num, replace=False)
    np.save(args.out, indices)

if __name__=="__main__":
    main()
