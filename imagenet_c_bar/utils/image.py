# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np



#def smoothstep(low, high, x):
#    return np.clip(3 * (x ** 2) - 2 * (x ** 3), 0, 1) * (high - low) + low

def smoothstep(low, high, x):
    x = np.clip(x, low, high)
    x = (x - low) / (high - low)
    return np.clip(3 * (x ** 2) - 2 * (x ** 3), 0, 1)


def bilinear_interpolation(image, point):
    l = int(np.floor(point[0]))
    u = int(np.floor(point[1]))
    r, d = l+1, u+1
    lu = image[l,u,:] if l >= 0 and l < image.shape[0]\
            and u >= 0 and u < image.shape[1] else np.array([0,0,0])
    ld = image[l,d,:] if l >= 0 and l < image.shape[0]\
            and d >= 0 and d < image.shape[1] else np.array([0,0,0])
    ru = image[r,u,:] if r >= 0 and r < image.shape[0]\
            and u >= 0 and u < image.shape[1] else np.array([0,0,0])
    rd = image[r,d,:] if r >= 0 and r < image.shape[0]\
            and d >= 0 and d < image.shape[1] else np.array([0,0,0])
    al = lu * (1.0 - point[1] + u) + ld * (1.0 - d + point[1])
    ar = ru * (1.0 - point[1] + u) + rd * (1.0 - d + point[1])
    out = al * (1.0 - point[0] + l) + ar * (1.0 - r + point[0])
    return out
