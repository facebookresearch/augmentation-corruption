# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Augmentation
import numpy as np

class Cifar10CropAndFlip(Augmentation):

    def sample_parameters(self):
        crop_pos = np.random.randint(low=-4, high=5, size=2)
        flip = (np.random.uniform() < 0.5)

        return {'crop_pos': crop_pos, 'flip': flip}

    def transform(self, image, crop_pos, flip):
        image = np.pad(image, ((4,4),(4,4),(0,0)))
        pos = crop_pos+4
        image = image[pos[0]:pos[0]+self.im_size,pos[1]:pos[1]+self.im_size,:]
        if flip:
            image = np.ascontiguousarray(image[:,::-1,:])
        return image.astype(np.uint8)
