# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Augmentation
from .utils.image import bilinear_interpolation
from .utils.severity import float_parameter, int_parameter, sample_level
from scipy.ndimage import shift, zoom, grey_erosion, grey_dilation
import numpy as np
from PIL import Image
from scipy.ndimage import rotate

class Scatter(Augmentation):

    tags = ['blur', 'filterpedia', 'scatter']
    name = 'scatter'

    def sample_parameters(self):
        seed = np.random.uniform(low=0.0, high=10.0)
        radius = float_parameter(sample_level(self.severity,self.max_intensity),  self.im_size/10)

        return {'seed' : seed, 'radius' : radius}

    def transform(self, image, seed, radius):

        def noise(x, y, seed):
            i, j = np.sin(x * seed), np.cos(y * seed)
            return (np.sin(12.9898*i + 78.233*j) * 43758.5453) % 1

        def warp_kernel(x, y, seed, radius):
            x_offset = radius * (-1.0 + noise(x, y, seed) * 2)
            y_offset = radius * (-1.0 + noise(y, x, seed) * 2)
            x_new = min(max(0, x+x_offset), self.im_size-1)
            y_new = min(max(0, y+y_offset), self.im_size-1)
            return y_new, x_new

        out = np.array([[bilinear_interpolation(image, warp_kernel(x, y, seed, radius))\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return out.astype(np.uint8)

class ChromaticAbberation(Augmentation):

    tags = ['blur', 'color', 'chromatic_abberation']
    name = 'chromatic_abberation'

    def sample_parameters(self):
#        shifts = np.array([int_parameter(sample_level(self.severity,self.max_intensity), self.im_size / 6)\
#                for i in range(6)]).reshape(3,2)
        angles = np.random.uniform(low=0, high=2*np.pi, size=3)
        dists = np.array([float_parameter(sample_level(self.severity, self.max_intensity), self.im_size / 10)\
                for i in range(3)])
        shifts = np.array([[np.cos(a)*d, np.sin(a)*d] for a, d in zip(angles, dists)])
#        flip = np.random.choice([-1,1], size=(3,2))
#        shifts = shifts * flip
        return { 'rgb_shifts' : shifts}

    def transform(self, image, rgb_shifts):
#        max_pad = np.max(np.abs(rgb_shifts))
#        image_padded = np.pad(image, [(max_pad, max_pad), (max_pad, max_pad), (0,0)])
        out = image.copy()
        for i in range(3):
            out[:,:,i] = shift(image[:,:,i], rgb_shifts[i], prefilter=False)
        #h, w, _ = image.shape
        #for i in range(3):
        #    out[:,:,i] = image_padded[max_pad+rgb_shifts[i,0]:max_pad+h+rgb_shifts[i,0],max_pad+rgb_shifts[i,1]:max_pad+w+rgb_shifts[i,1],i]
        return out

    def convert_to_numpy(self, params):
        return params['rgb_shifts'].flatten()

    def convert_from_numpy(self, numpy_record):
        return {'rgb_shifts' : numpy_record.reshape(3,2).astype(np.int)}

class TransverseChromaticAbberation(Augmentation):

    tags = ['blur', 'color', 'pil', 'transverse_chromatic_abberation']
    name = 'transverse_chromatic_abberation'

    def sample_parameters(self):
        scales = np.array([float_parameter(sample_level(self.severity,self.max_intensity), 0.5)\
                for i in range(3)])
        scale = float_parameter(sample_level(self.severity, self.max_intensity), 0.5)
        scales = np.array([1.0, 1.0+scale/2, 1.0+scale])
        scales = scales[np.random.permutation(3)]

        #zerod = np.random.randint(low=0, high=3)
        #scales[zerod] = 0.0
        #flip = np.random.choice([-1, 1], size=3)
        #scales = flip * scales
        #scales = 2 ** scales
        return { 'scales' : scales }

    def transform(self, image, scales):
        out = image.copy()
        for c in range(3):
            zoomed = zoom(image[:,:,c], scales[c], prefilter=False)
            edge = (zoomed.shape[0]-self.im_size)//2
            out[:,:,c] = zoomed[edge:edge+self.im_size, edge:edge+self.im_size]
        return out.astype(np.uint8)
        '''
        image = Image.fromarray(image)
        channel_list = []
        for i, channel in enumerate(image.getbands()):
            im = image.getchannel(channel)
            affine = np.array([[scales[i], 0, (1-scales[i])*self.im_size/2], [0, scales[i], (1-scales[i])*self.im_size/2]])
            im = im.transform((self.im_size, self.im_size), Image.AFFINE, affine.flatten())
            channel_list.append(im)
        out = Image.merge("RGB", channel_list)
        return np.array(out).astype(np.uint8)
        '''
            
    def convert_to_numpy(self, params):
        return params['scales'].flatten()

    def convert_from_numpy(self, numpy_record):
        return {'scales' : numpy_record}

class HomogeneousColorBlur(Augmentation):

    tags = ['blur', 'filterpedia', 'homogenous_color_blur', 'slow', 'impractically_slow']
    name = 'homogeneous_color_blur'

    def sample_parameters(self):
        radius = int_parameter(sample_level(self.severity,self.max_intensity), self.im_size/10)
        threshold = np.random.uniform(low=0.2, high=0.21)

        return { 'radius' : radius, 'threshold' : threshold }

    def transform(self, image, radius, threshold):
        
        def kernel(point, image, radius, threshold):
            this_val = image[point[0],point[1],:]
            acc = np.zeros(3)
            n = 0
            for x in np.arange(-radius, radius+1):
                for y in np.arange(-radius, radius+1):
                    x_pos = point[0]+x
                    y_pos = point[1]+y
                    if x_pos < 0 or x_pos >= self.im_size or y_pos < 0 or y_pos >= self.im_size:
                        continue
                    offset_val = image[x_pos,y_pos,:]
                    dist_mul = 1 if radius >= np.sqrt(x**2+y**2) else 0
                    color_mul = 1 if 255*threshold >= np.sqrt(np.sum((this_val-offset_val)**2)) else 0
                    acc += offset_val * dist_mul * color_mul
                    n += dist_mul * color_mul
            return acc / n

        out = np.array([[kernel(np.array([y,x]), image, radius, threshold)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return out.astype(np.uint8)


class Erosion(Augmentation):

    tags = ['blur']
    name = 'erosion'

    def sample_parameters(self):
        r2 = float_parameter(sample_level(self.severity, self.max_intensity), (5**2-1.5**2)) + 1.5**2
        radius = np.sqrt(r2) 

        return {'radius' : radius}

    def transform(self, image, radius):

        mask = np.zeros((np.ceil(2*radius).astype(np.uint8), np.ceil(2*radius).astype(np.uint8)))
        center = np.array([radius, radius])
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                mask[x,y] = 1 if np.linalg.norm(np.array([x,y])-center) <= radius else 0
        if np.max(mask) == 0:
            return image
        out = image.copy()
        for c in range(3):
            out[:,:,c] = grey_erosion(out[:,:,c], footprint=mask)
        return out 


class Dilation(Augmentation):

    tags = ['blur']
    name = 'dilation'

    def sample_parameters(self):
        r2 = float_parameter(sample_level(self.severity, self.max_intensity), (5**2-1.5**2)) + 1.5**2
        radius = np.sqrt(r2) 

        return {'radius' : radius}

    def transform(self, image, radius):

        mask = np.zeros((np.ceil(2*radius).astype(np.uint8), np.ceil(2*radius).astype(np.uint8)))
        center = np.array([radius, radius])
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                mask[x,y] = 1 if np.linalg.norm(np.array([x,y])-center) <= radius else 0
        if np.max(mask) == 0:
            return image
        out = image.copy()
        for c in range(3):
            out[:,:,c] = grey_dilation(out[:,:,c], footprint=mask)
        return out 

class CircularMotionBlur(Augmentation):

    tags = ['blur']
    name = 'circular_motion_blur'

    def sample_parameters(self):
        amount = float_parameter(sample_level(self.severity, self.max_intensity),15)

        return {'amount' : amount}

    def transform(self, image, amount):

        num = 21
        factors = []
        rotated = []
        image = image.astype(np.float32) / 255
        for i in range(num):
            angle = (2*i/(num-1) - 1) * amount
            rotated.append(rotate(image, angle, reshape=False))
            factors.append(np.exp(- 2*(2*i/(num-1)-1)**2))
        out = np.zeros_like(image)
        for i, f in zip(rotated, factors):
            out += f * i
        out /= sum(factors)
        return np.clip(out*255, 0, 255).astype(np.uint8)
        
