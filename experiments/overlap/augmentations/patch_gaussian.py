# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Augmentation
from math import floor, ceil
import numpy as np

class Gaussian(Augmentation):
    name = 'pg_gaussian'
    tags = ['float_return']

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        sigma = np.random.uniform(low=0, high=self.severity/10)\
                if not self.max_intensity else self.severity/10

        return {'seed': seed, 'sigma': sigma}

    def transform(self, image, seed, sigma):
        random_state = np.random.RandomState(seed=seed)
        noise = random_state.randn(self.im_size, self.im_size, 3)
        image = image.astype(np.float32) / 255
        image = np.clip(image+sigma*noise, 0, 1)
        return image * 255



class PatchGaussian(Augmentation):

    name = 'patch_gaussian'
    tags = ['float_return', 'additional_parameters']

    def __init__(self, severity, im_size, record=False, max_intensity=False, sigma=1.0, width=None):
        super(PatchGaussian, self).__init__(severity, im_size, record, max_intensity)
        self.sigma = sigma
        self.width = width if width is not None else self.im_size

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        crop_pos = np.random.randint(low=0, high=self.im_size, size=2)
        width = np.random.uniform(low=0, high=self.width)\
                if not self.max_intensity else self.width
        sigma = np.random.uniform(low=0, high=self.sigma)

        return {'seed': seed, 'crop_pos': crop_pos, 'sigma': sigma, 'width': width}

    def transform(self, image, seed, crop_pos, sigma, width):

        random_state = np.random.RandomState(seed=seed)
        noise = random_state.randn(self.im_size, self.im_size, 3)
        noise *= sigma
        mask = np.zeros((self.im_size, self.im_size))
        l = int(max(0, crop_pos[0]-floor(width/2)))
        r = int(min(self.im_size, crop_pos[0]+ceil(width/2)))
        u = int(max(0, crop_pos[1]-floor(width/2)))
        d = int(min(self.im_size, crop_pos[1]+ceil(width/2)))
        mask[l:r,u:d] = 1.0
        mask = mask.reshape(self.im_size, self.im_size, 1)
        image = image.astype(np.float32) / 255
        image = image + mask * noise
        image = np.clip(image, 0, 1)
        return image * 255
        




