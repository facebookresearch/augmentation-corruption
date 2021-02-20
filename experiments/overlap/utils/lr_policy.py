# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

class Cosine(object):
    def __init__(self, base_lr, max_epoch):
        self.base_lr = base_lr
        self.max_epoch = max_epoch

    def __call__(self, cur_epoch):
        return 0.5 * self.base_lr * (1.0 + np.cos(np.pi * cur_epoch / self.max_epoch))

class Steps(object):
    def __init__(self, base_lr, lr_mult, steps):
        self.base_lr = base_lr
        self.lr_mult = lr_mult
        self.steps = steps

    def __call__(self, cur_epoch):
        ind = [i for i, s in enumerate(self.steps) if cur_epoch >= s][-1]
        return self.base_lr * self.lr_mult ** ind
