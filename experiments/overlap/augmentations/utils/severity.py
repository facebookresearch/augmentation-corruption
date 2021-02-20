# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

def int_parameter(level, maxval):
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  return float(level) * maxval / 10.


def sample_level(n, fixed=False):
  if fixed:
      return n
  return np.random.uniform(low=0.1, high=n)
