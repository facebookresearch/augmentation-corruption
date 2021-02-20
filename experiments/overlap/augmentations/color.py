# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Augmentation
import numpy as np
from .utils.severity import int_parameter, float_parameter, sample_level
from .utils.image import smoothstep
from skimage.color import rgb2hsv, hsv2rgb

class BleachBypass(Augmentation):

    tags = ['color', 'filterpedia', 'bleach_bypass']
    name = 'bleach_bypass'

    def sample_parameters(self):
        amount = float_parameter(sample_level(self.severity,self.max_intensity), 1.0)
        return { 'amount' : amount }

    def transform(self, image, amount):
        vals = np.array([0.2126, 0.7152, 0.0722]).reshape(1,1,3)
        luma = np.sum(image*vals, axis=2, keepdims=True)/255
        l = np.clip(10.0 * (luma - 0.45), 0, 1.0)
        result1 = 2 * image * luma / 255
        result2 = 1.0 - 2.0 * (1.0 - luma) * (1.0 - image /255)
        out = ((1-l) * result1 + l * result2) * 255

        return ((1-amount) * image + amount * out).astype(np.uint8)

class Technicolor(Augmentation):

    tags = ['color', 'filterpedia', 'technicolor']
    name = 'technicolor'

    def sample_parameters(self):
        amount = float_parameter(sample_level(self.severity,self.max_intensity), 1.0)
        return { 'amount' : amount }

    def transform(self, image, amount):

        redmatte = 1.0 - (image[:,:,0]/255 - ((image[:,:,1]/2+image[:,:,2]/2))/255)
        greenmatte = 1.0 - (image[:,:,1]/255 - ((image[:,:,0]/2+image[:,:,2]/2))/255)
        bluematte = 1.0 - (image[:,:,2]/255 - ((image[:,:,0]/2+image[:,:,1]/2))/255)

        red = greenmatte * bluematte * image[:,:,0].astype(np.float32)
        green = redmatte * bluematte * image[:,:,1].astype(np.float32)
        blue = redmatte * greenmatte * image[:,:,2].astype(np.float32)

        new_color = np.stack((red, green, blue), axis=2)
        return ((1-amount) * image + amount * new_color).astype(np.uint8)

class Pseudocolor(Augmentation):

    tags = ['color', 'filterpedia', 'pseudocolor']
    name = 'pseudocolor'

    def sample_parameters(self):
        smoothness = np.random.uniform(low=0.25, high=0.75)
        color0 = np.random.randint(low=0, high=255, size=3).astype(np.uint8)
        color1 = np.random.randint(low=0, high=255, size=3).astype(np.uint8)
        color2 = np.random.randint(low=0, high=255, size=3).astype(np.uint8)
        color3 = np.random.randint(low=0, high=255, size=3).astype(np.uint8)
        color4 = np.random.randint(low=0, high=255, size=3).astype(np.uint8)
        amount = float_parameter(sample_level(self.severity,self.max_intensity), 0.5)

        return { 'smoothness' : smoothness, 'color0' : color0, 'color1': color1,
                'color2': color2, 'color3' : color3, 'color4' : color4, 'amount' : amount }

    def transform(self, image, color0, color1, color2, color3, color4, smoothness, amount):

        color0 = color0.astype(np.uint8)
        color1 = color1.astype(np.uint8)
        color2 = color2.astype(np.uint8)
        color3 = color3.astype(np.uint8)
        color4 = color4.astype(np.uint8)
        def get_color(color0, color1, edge0, edge1, luma, smoothness):
            smooth_color = color0 + ((color1 - color0) * smoothstep(edge0, edge1, luma))
            a = 4.0 * (luma - edge0)
            linear_color = (1 - a) * color0 + a * color1
            return (1 - smoothness) * linear_color + smoothness * smooth_color

        vals = np.array([0.2126, 0.7152, 0.0722]).reshape(1,1,3)
        luma = np.sum(image.astype(np.float32)*vals, axis=2, keepdims=True)/255

        c1 = get_color(color0, color1, 0.0, 0.25, luma, smoothness)
        c2 = get_color(color1, color2, 0.25, 0.50, luma, smoothness)
        c3 = get_color(color2, color3, 0.5, 0.75, luma, smoothness)
        c4 = get_color(color3, color4, 0.75, 1.0, luma, smoothness)
        out = (luma < 0.25) * c1 + ((luma >= 0.25)&(luma < 0.5)) * c2\
                + ((luma >= 0.5)&(luma < 0.75)) * c3 + (luma >= 0.75) * c4
        return np.clip((1 - amount) * image + amount * out, 0, 255).astype(np.uint8)
        
    def convert_to_numpy(self, params):
        colors = []
        for i in range(5):
            colors.extend(params['color'+str(i)].tolist())
        return np.array([params['smoothness']] + colors + [params['amount']])

    def convert_from_numpy(self, numpy_record):
        params = {'smoothness' : numpy_record[0], 'amount' : numpy_record[16]}
        for i in range(5):
            params['color'+str(i)] = numpy_record[1+3*i:1+3*(i+1)]
        return params

class HueShift(Augmentation):

    tags = ['color']
    name = 'hue_shift'

    def sample_parameters(self):
        amount = float_parameter(sample_level(self.severity, self.max_intensity), 0.5)
        if np.random.uniform() < 0.5:
            amount *= -1
        return {'amount' : amount}

    def transform(self, image, amount):
        
        hsv_image = rgb2hsv(image.astype(np.float32)/255)
        hsv_image[:,:,0] += (amount % 1.0)

        return np.clip(hsv2rgb(hsv_image)*255, 0, 255).astype(np.uint8)

class ColorDither(Augmentation):

    tags = ['color']
    name = 'color_dither'

    def sample_parameters(self):
        #factor = float_parameter(sample_level(self.severity, self.max_intensity),6.0)+1.0
        #levels = int(256 / (2**factor))
        levels = int_parameter(sample_level(self.severity, self.max_intensity),10)
        levels = 14-levels

        return {'levels' : levels}

    def transform(self, image, levels):

        index = 0
        color_map = [int(255 * i / (levels -1)) for i in range(levels)]
        div = [int(levels*i / 256) for i in range(256)]
        out = np.zeros_like(image)
        image_copy = image.copy()
        m = np.array([[0,0,0],[0,0,7],[3,5,1]])
        

        for y in range(self.im_size):
            reverse = ((y % 1) == 1)
            if reverse:
                index = y*self.im_size + self.im_size - 1
                direction = -1
            else:
                index = y*self.im_size
                direction = 1

            for x in range(self.im_size):
                curr_val = image_copy[index//self.im_size, index%self.im_size,:]

                new_val = np.array([color_map[div[c]] for c in curr_val])
                out[index//self.im_size, index%self.im_size,:] = new_val

                e = curr_val - new_val

                for i in [-1,0,1]:
                    iy = y+i
                    if iy > 0 and iy < self.im_size:
                        for j in [-1,0,1]:
                            jx = x+j
                            if jx > 0 and jx < self.im_size:
                                if reverse:
                                    w = m[(i+1),-j+1]
                                else:
                                    w = m[(i+1),j+1]
                                if w != 0:
                                    k = index - j if reverse else index + j
                                    curr_val = image_copy[k//self.im_size, k%self.im_size,:].astype(np.float32)
                                    curr_val = np.clip(curr_val + e * w/np.sum(m),0,255).astype(np.uint8)
                                    image_copy[k//self.im_size,k%self.im_size,:] = curr_val
                index += direction
        return np.clip(out, 0, 255).astype(np.uint8)



