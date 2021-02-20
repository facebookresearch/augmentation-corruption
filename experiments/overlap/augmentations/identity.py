# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Augmentation

class Identity(Augmentation):

    tags = ["identity"]
    name = ['identity']

    def __init__(self, severity=None, record=False, **kwargs):
        super(Identity, self).__init__(severity, record, **kwargs)

    def sample_parameters(self):
        return {}

    def transform(self, image):
        return image
