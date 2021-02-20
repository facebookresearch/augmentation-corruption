# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import Image
import torch

class PilToNumpy(object):
    def __init__(self, as_float=False, scaled_to_one=False):
        self.as_float = as_float
        self.scaled_to_one = scaled_to_one
        assert (not scaled_to_one) or (as_float and scaled_to_one),\
                "Must output a float if rescaling to one."

    def __call__(self, image):
        if not self.as_float:
            return np.array(image).astype(np.uint8)
        elif notself.scaled_to_one:
            return np.array(image).astype(np.float32)
        else:
            return np.array(image).astype(np.float32) / 255

class NumpyToPil(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return Image.fromarray(image)

class NumpyToTensor(object):
    def __init__(self, HWC_to_CHW=True, bit_to_float=True):
        self.HWC_to_CHW = HWC_to_CHW
        self.bit_to_float = bit_to_float
        pass

    def __call__(self, image):
        image = image.astype(np.float32)
        if self.bit_to_float:
            image /= 255
        if self.HWC_to_CHW:
            image = image.transpose(2,0,1)
        return torch.Tensor(image)
