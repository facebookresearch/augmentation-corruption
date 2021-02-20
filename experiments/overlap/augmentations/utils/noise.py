# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

class PerlinNoiseGenerator(object):
    def __init__(self, random_state=None):
        self.rand = np.random if random_state is None else random_state

        B = 256
        N = 16*256

        def normalize(arr):
            return arr / np.linalg.norm(arr)

        self.p = np.arange(2*B+2)
        self.g = np.array([normalize((random_state.randint(low=0, high=2**31, size=2) % (2*B) - B )/ B)\
                for i in range(2*B+2)])


        for i in np.arange(B-1,-1,-1):
            k = self.p[i]
            j = self.rand.randint(low=0, high=2**31) % B
            self.p[i] = self.p[j]
            self.p[j] = k

        for i in range(B+2):
            self.p[B+i] = self.p[i]
            self.g[B+i,:] = self.g[i,:]
        self.B = B
        self.N = N


    def s_curve(t):
        return t**2 * (3.0 - 2.0 * t)

    def noise(self, x, y):

        t = x + self.N
        bx0 = int(t) % self.B
        bx1 = (bx0+1) % self.B
        rx0 = t % 1
        rx1 = rx0 - 1.0

        t = y + self.N
        by0 = int(t) % self.B
        by1 = (by0+1) % self.B
        ry0 = t % 1
        ry1 = ry0 - 1.0

        i = self.p[bx0]
        j = self.p[bx1]

        b00 = self.p[i + by0]
        b10 = self.p[j + by0]
        b01 = self.p[i + by1]
        b11 = self.p[j + by1]

        sx = PerlinNoiseGenerator.s_curve(rx0)
        sy = PerlinNoiseGenerator.s_curve(ry0)

        u = rx0 * self.g[b00,0] + ry0 * self.g[b00,1]
        v = rx1 * self.g[b10,0] + ry0 * self.g[b10,1]
        a = u + sx * (v - u)

        u = rx0 * self.g[b01,0] + ry1 * self.g[b01,1]
        v = rx1 * self.g[b11,0] + ry1 * self.g[b11,1]
        b = u + sx * (v - u)

        return 1.5 * (a + sy * (b - a))

    def turbulence(self, x, y, octaves):
        t = 0.0
        f = 1.0
        while f <= octaves:
            t += np.abs(self.noise(f*x, f*y)) / f
            f = f * 2
        return t




        
