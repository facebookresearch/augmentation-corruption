# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import floor, ceil
import numpy as np
from .base import Augmentation
from .utils.severity import int_parameter, sample_level, float_parameter
from skimage.draw import line_aa
from scipy.fftpack import ifft2

class CutOut(Augmentation):

    tags = ['autoaugment', 'cutout']
    name = 'cutout'

    def sample_parameters(self):
        center = np.random.randint(low=0, high=self.im_size, size=2)
        size = int_parameter(sample_level(self.severity, self.max_intensity), 15)+1
        return {'center' : center, 'size' : size}

    def transform(self, image, center, size):
        out = image.copy()
        lu = np.clip(center-floor(size/2), 0, self.im_size)
        rd = np.clip(center+ceil(size/2), 0, self.im_size)
        out[lu[0]:rd[0],lu[1]:rd[1],:] = [128,128,128]
        return out

    def convert_to_numpy(self, params):
        return np.array(params['center'].tolist() + [params['size']])

    def convert_from_numpy(self, numpy_record):
        return {'center' : numpy_record[0:2].astype(np.int), 'size' : numpy_record[2]}
    
'''
class CheckerBoardCutOut(Augmentation):

    tags = ['checkerboard_cutout']
    name = 'checkerboard_cutout'

    def sample_parameters(self):
        if self.max_intensity:
            size = max(1, int(self.severity))
        else:
            size = np.random.randint(low=1, high=max(1,int(self.severity))+1)
        offset = np.random.randint(low=0, high=size+1, size=2)

        return { 'offset' : offset, 'size' : size}

    def transform(self, image, offset, size):
        out = image.copy()
        num = self.im_size // size + 2
        for i in range(num):
            for j in range(num):
                if (i+j) % 2 == 0:
                    continue
                l = np.clip((i-1)*size+offset[0],0,self.im_size)
                r = np.clip((i)*size+offset[0],0,self.im_size)
                u = np.clip((j-1)*size+offset[1],0,self.im_size)
                d = np.clip((j)*size+offset[1],0,self.im_size)
                out[l:r,u:d,:] = [128,128,128]
        return out

    def convert_to_numpy(self, params):
        return np.array(params['offset'].tolist() + [params['size']])

    def convert_from_numpy(self, numpy_record):
        return {'offset' : numpy_record[0:2].astype(np.int), 'size' : numpy_record[2].astype(np.int)}
'''
'''
class CheckerBoardCutOut(Augmentation):

    tags = ['obscure']
    name = 'checkerboard_cutout'

    def sample_parameters(self):
        angle = np.random.uniform(low=0, high=2*np.pi)
        #scales = np.array([0.5, 0.5]) * self.im_size
        scales = np.array([float_parameter(sample_level(self.severity, self.max_intensity), 1.0)\
                for i in range(2)])
        scales = np.maximum((1.1 - scales) * 0.25 * self.im_size, 1)

        return {'angle' : angle, 'scales' : scales}

    def transform(self, image, scales, angle):
        
        def mask_kernel(point, scales, angle):
            nx = (np.cos(angle) * point[0] + np.sin(angle) * point[1]) / scales[0]
            ny = (-np.sin(angle) * point[0] + np.cos(angle) * point[1]) / scales[1]
            return int(nx % 2) != int(ny % 2)  

        out = np.array([[image[y,x,:] if mask_kernel([y,x], scales, angle) else np.array([128,128,128])\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip(out, 0, 255).astype(np.uint8)
'''

class CheckerBoardCutOut(Augmentation):

    tags = ['obscure']
    name = 'checkerboard_cutout'

    def sample_parameters(self):
        angle = np.random.uniform(low=0, high=2*np.pi)
        scales = np.maximum(np.random.uniform(low=0.1, high=0.25) * self.im_size, 1)
        scales = (scales, scales)
        fraction = float_parameter(sample_level(self.severity, self.max_intensity), 1.0)
        seed = np.random.randint(low=0, high=2**32)

        return {'angle' : angle, 'scales' : scales, 'fraction' : fraction, 'seed' : seed}

    def transform(self, image, scales, angle, fraction, seed):
        random_state = np.random.RandomState(seed=seed)
        grid = random_state.uniform(size=(int(4*self.im_size//scales[0]), int(4*self.im_size//scales[1]))) < fraction
        
        def mask_kernel(point, scales, angle, grid):
            nx = (np.cos(angle) * point[0] + np.sin(angle) * point[1]) / scales[0]
            ny = (-np.sin(angle) * point[0] + np.cos(angle) * point[1]) / scales[1]
            return (int(nx % 2) != int(ny % 2)) or not grid[int(nx),int(ny)]

        out = np.array([[image[y,x,:] if mask_kernel([y,x], scales, angle, grid) else np.array([128,128,128])\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip(out, 0, 255).astype(np.uint8)


class Lines(Augmentation):

    tags = ['obscure']
    name = 'lines'

    def sample_parameters(self):
        length = 1.0
        density = float_parameter(sample_level(self.severity, self.max_intensity), 1.0)
        angle = np.random.uniform(low=0.0, high=2*np.pi)
        angle_variation = np.random.uniform(low=0.1, high=1.0)
        seed = np.random.randint(low=0, high=2**32)

        return {'length' : length, 'density' : density, 'angle' : angle, 'angle_variation' : angle_variation, 'seed' : seed}

    def transform(self, image, length, density, angle, angle_variation, seed):

        num_lines = int(density * self.im_size)
        l = length * self.im_size
        random_state = np.random.RandomState(seed=seed)
        out = image.copy()
        for i in range(num_lines):
            x = self.im_size * random_state.uniform()
            y = self.im_size * random_state.uniform()
            a = angle + 2 * np.pi * angle_variation * (random_state.uniform() - 0.5)
            s = np.sin(a) * l
            c = np.cos(a) * l
            #x1 = max(min(int(x-c), self.im_size-1), 0)
            #x2 = max(min(int(x+c), self.im_size-1), 0)
            #y1 = max(min(int(y-s), self.im_size-1), 0)
            #y2 = max(min(int(y+s), self.im_size-1), 0)
            x1 = int(x-c)
            x2 = int(x+c)
            y1 = int(y-s)
            y2 = int(y+s)
            rxc, ryc, rval = line_aa(x1, y1, x2, y2)
            xc, yc, val = [], [], []
            for rx, ry, rv in zip(rxc, ryc, rval):
                if rx >= 0 and ry >= 0 and rx < self.im_size and ry < self.im_size:
                    xc.append(rx)
                    yc.append(ry)
                    val.append(rv)
            xc, yc, val = np.array(xc, dtype=np.int), np.array(yc, dtype=np.int), np.array(val)
            out[xc, yc, :] = (1.0 - val.reshape(-1,1)) * out[xc, yc, :].astype(np.float32) + val.reshape(-1,1)*128
        return out.astype(np.uint8)

class RandomSample(Augmentation):

    tags = ['obscure']
    name = 'random_sample'

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        density = 1.0 - float_parameter(sample_level(self.severity, self.max_intensity), 0.8)

        return {'density' : density, 'seed' : seed}

    def transform(self, image, density, seed):

        random_state = np.random.RandomState(seed=seed)
        num = int(density * self.im_size ** 2)
        out = np.zeros_like(image)

        #for i in range(num):
        #    point = np.random.randint(low=0, high=self.im_size, size=2)
        #    out[point[0], point[1], :] = image[point[0], point[1], :]
        indices = random_state.choice(np.arange(self.im_size**2), size=num, replace=False)
        for idx in indices:
            out[idx//self.im_size, idx % self.im_size, :] = image[idx//self.im_size, idx % self.im_size, :]

        return out


class BlueNoiseSample(Augmentation):

    tags = ['obscure']
    name = 'blue_noise_sample'

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        threshold = float_parameter(sample_level(self.severity, self.max_intensity), 3.0) - 2.5

        return {'seed' : seed, 'threshold' : threshold}

    def transform(self, image, seed, threshold):
        random_state = np.random.RandomState(seed=seed)

        center = self.im_size / 2
        power = np.array([[np.linalg.norm(np.array([x,y])-center)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        #power = power / self.im_size

        phases = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size, self.im_size//2))
        if self.im_size % 2 == 0:
            phases = np.concatenate((phases, phases[::-1,::-1]), axis=1)
        else:
            center_freq = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size//2, 1))
            center_freq = np.concatenate((center_freq, np.array([[0.0]]), center_freq[::-1,:]), axis=0)
            phases = np.concatenate((phases, center_freq, phases[::-1,::-1]), axis=1)
        fourier_space_noise = power * (np.cos(phases) + np.sin(phases) * 1j)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=0)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=1)


        noise = np.real(ifft2(fourier_space_noise))
        noise = noise / np.std(noise)
        mask = noise > threshold
        out = image * mask.reshape(self.im_size, self.im_size, 1)


        return np.clip(out, 0, 255).astype(np.uint8)
