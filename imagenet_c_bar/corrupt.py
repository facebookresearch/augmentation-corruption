# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from math import floor, ceil
from PIL import Image
from scipy.fftpack import ifft2
from scipy.ndimage import gaussian_filter, rotate, shift, zoom
from skimage.draw import line_aa
from skimage.color import rgb2hsv, hsv2rgb
from utils.image import bilinear_interpolation, smoothstep
from utils.perlin_noise import PerlinNoiseGenerator
from base import Transform
import abc

def int_parameter(level, maxval):
  return int(level * maxval / 10)

def float_parameter(level, maxval):
  return float(level) * maxval / 10.

class SingleFrequencyGreyscale(Transform):

    name = 'single_frequency_greyscale'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        freq_mag = np.random.uniform(low=-np.pi, high=np.pi)
        freq_2 = np.random.uniform(low=-abs(freq_mag), high=abs(freq_mag))
        freq = np.array([freq_mag, freq_2])[np.random.permutation(2)]
        phase = np.random.uniform(low=0, high=2*np.pi)
        intensity = float_parameter(self.severity, 196)
        return { 'freq' : freq, 'phase' : phase, 'intensity' : intensity}

    def transform(self, image, freq, phase, intensity):
        noise = np.array([[np.sin(x * freq[0] + y * freq[1] + phase)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        noise = np.stack((noise, noise, noise), axis=2)

        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)

    def convert_to_numpy(self, params):
        return np.array(params['freq'].tolist() + [params['phase'], params['intensity']])

    def convert_from_numpy(self, numpy_record):
        return {'freq' : numpy_record[0:2],
                'phase' : numpy_record[2],
                'intensity' : numpy_record[3]
                }

class CocentricSineWaves(Transform):

    name = 'cocentric_sine_waves'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        offset = np.random.uniform(low=0, high=self.im_size, size=2)
        freq = np.random.uniform(low=0, high=10)
        amplitude = np.random.uniform(low=0, high=self.im_size/10)
        ring_width = np.random.uniform(low=0, high=self.im_size/10)
        intensity = [float_parameter(self.severity, 128) for i in range(3)]

        return { 'offset' : offset,
                 'freq' : freq,
                 'amplitude' : amplitude,
                 'ring_width' : ring_width,
                 'intensity' : intensity
                }

    def transform(self, image, offset, freq, amplitude, ring_width, intensity):

        def calc_intensity(x, y, x0, y0, freq, amplitude, ring_width):
            angle = np.arctan2(x-x0, y-y0) * freq
            distance = ((np.sqrt((x-x0)**2 + (y-y0)**2) + np.sin(angle) * amplitude) % ring_width) / ring_width
            distance -= 1/2
            return distance

        noise = np.array([[calc_intensity(x, y, offset[0], offset[1], freq, amplitude, ring_width)\
                    for x in range(self.im_size)] for y in range(self.im_size)])
        noise = np.stack((intensity[0] * noise, intensity[1] * noise, intensity[2] * noise), axis=2)

        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def convert_to_numpy(self, params):
        return np.array(params['offset'].tolist() + [params['freq'], params['amplitude'], params['ring_width']] + params['intensity'])

    def convert_from_numpy(self, numpy_record):
        return {'offset' : numpy_record[0:2].tolist(),
                'freq' : numpy_record[2],
                'amplitude' : numpy_record[3],
                'ring_width' : numpy_record[4],
                'intensity' : numpy_record[4:7].tolist()
                }
        

class PlasmaNoise(Transform):

    name = 'plasma_noise'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        time = np.random.uniform(low=0.0, high=6*np.pi)
        iterations = np.random.randint(low=4, high=7)
        sharpness = np.random.uniform(low=0.5, high=1.0)
        scale = np.random.uniform(low=0.075, high=0.2) * self.im_size
        intensity = float_parameter(self.severity,64)
        return {'time' : time, 'iterations' : iterations, 'sharpness' : sharpness,
                'scale' : scale, 'intensity' : intensity}

    def transform(self, image, time, iterations, sharpness, scale, intensity):

        def kernel(x, y, rand, iters, sharp, scale):
            x /= scale
            y /= scale
            i = np.array([1.0, 1.0, 1.0, 0.0])
            for s in range(iters):
                r = np.array([np.cos(y * i[0] - i[3] + rand / i[1]), np.sin(x * i[0] - i[3] + rand / i[1])]) / i[2]
                r += np.array([-r[1],r[0]]) * 0.3
                x += r[0]
                y += r[1]
                i *= np.array([1.93, 1.15, (2.25 - sharp), rand * i[1]])
            r = np.sin(x - rand)
            b = np.sin(y + rand)
            g = np.sin((x + y + np.sin(rand))*0.5)
            return [r,g,b]


        noise = np.array([[kernel(x,y, time, iterations, sharpness, scale)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip((1-intensity/255) * image + intensity * noise, 0, 255).astype(np.uint8)


class VoronoiNoise(Transform):

    name = 'voronoi_noise'
    tags = ['new_corruption']

    def sample_parameters(self):
        seed = np.random.uniform(low=0, high=10)
        density = np.random.uniform(low=0.5, high=0.9)
        size = np.random.uniform(low=0.05, high=0.2) * self.im_size
        intensity = float_parameter(self.severity,255)
        if np.random.uniform() > 0.5:
            intensity = -intensity

        return {'seed' : seed, 'density' : density, 'size' : size, 'intensity' : intensity}


    def transform(self, image, size, seed, density, intensity):

        def voronoi_hash(v, time):
            m = np.array([[13.85, 47.77], [99.41, 88.48]])
            w = np.matmul(m,v)
            return (np.sin(w) * np.sqrt(w) * time * 0.0025) % 1

        def kernel(x, y, size, seed, density):
            v = np.array([[x],[y]]) / size + 1
            g = v // 1
            f = v % 1


            dist = 1.0
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    p = np.array([[i],[j]])
                    curr_dist = np.linalg.norm((p + voronoi_hash(g+p, seed) - f).flatten())
                    dist = min(curr_dist, dist)

            r = smoothstep(0, 1, dist * density)  - 0.5
            return r


        noise = np.array([[kernel(x,y, size, seed, density)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        noise = np.stack((noise, noise, noise), axis=2)
        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)

class CausticNoise(Transform):

    name = 'caustic_noise'
    tags = ['new_corruption']

    def sample_parameters(self):
        time = np.random.uniform(low=0.5, high=2.0)
        size = np.random.uniform(low=0.75, high=1.25) * self.im_size
        #size = self.im_size
        intensity = float_parameter(self.severity, 255)

        return { 'time' : time, 'size' : size, 'intensity' : intensity}

    def transform(self, image, time, size, intensity):

        def kernel(point, time, size):
            point = point / size
            p = (point % 1) * 6.28318530718 - 250

            i = p.copy()
            c = 1.0
            inten = 0.005

            for n in range(5):
                t = time * (1.0 - (3.5 / (n+1)))
                i = p + np.array([np.cos(t-i[0])+np.sin(t+i[1]),np.sin(t-i[1])+np.cos(t+i[0])])
                length = np.sqrt((p[0] / (np.sin(i[0]+t)/inten))**2 + (p[1] / (np.cos(i[1]+t)/inten))**2)
                c += 1.0/length

            c /= 5.0
            c = 1.17 - c ** 1.4
            color = np.clip(np.abs(c) ** 8.0, 0, 1) 
            return np.array([color, color, color])

        noise = np.array([[kernel(np.array([y,x]), time, size)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip(image + intensity  *  noise, 0, 255).astype(np.uint8)

class Sparkles(Transform):

    name = 'sparkles'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        centers = np.random.uniform(low=0, high=self.im_size, size=(5, 2))
        radii = np.array([float_parameter(self.severity, 0.1)\
                for i in range(5)]) * self.im_size
        amounts = np.array([50 for i in range(5)])
        color = np.array([255, 255, 255])
        randomness = 25
        seed = np.random.randint(low=0, high=2**32)
        nrays = np.random.randint(low=50, high=200, size=5)

        return {'centers' : centers, 'radii' : radii, 'color' : color, 'randomness' : randomness,
                'seed' : seed, 'nrays' : nrays, 'amounts' : amounts
                }

    def transform(self, image, centers, radii, nrays, amounts, color, randomness, seed):

        def kernel(point, value, center, radius, ray_lengths, amount, color):
            rays = len(ray_lengths)
            dp = point - center
            dist = np.linalg.norm(dp)
            angle = np.arctan2(dp[1], dp[0])
            d = (angle + np.pi) / (2 * np.pi) * rays
            i = int(d)
            f = d - i 

            if radius != 0:
                length = ray_lengths[i % rays] + f * (ray_lengths[(i+1) % rays] - ray_lengths[i % rays])
                g = length**2 / (dist**2 + 1e-4)
                g = g ** ((100 - amount) / 50.0)
                f -= 0.5
                f = 1 - f**2
                f *= g
            f = np.clip(f, 0, 1)
            return value + f * (color - value)

        random_state = np.random.RandomState(seed=seed)
        for center, rays, amount, radius in zip(centers, nrays, amounts, radii):
            ray_lengths = [max(1,radius + randomness / 100.0 * radius * random_state.randn())\
                for i in range(rays)]

            image = np.array([[kernel(np.array([y,x]), image[y,x,:].astype(np.float32), center, radius, ray_lengths, amount, color)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(image, 0, 255).astype(np.uint8)


class InverseSparkles(Transform):

    name = 'inverse_sparkles'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        center = np.random.uniform(low=0.25, high=0.75, size=2) * self.im_size
        radius = 0.25 * self.im_size
        amount = 100
        amount = float_parameter(self.severity, 65)
        amount = 100 - amount
        color = np.array([255, 255, 255])
        randomness = 25
        seed = np.random.randint(low=0, high=2**32)
        rays = np.random.randint(low=50, high=200)

        return {'center' : center, 'radius' : radius, 'color' : color, 'randomness' : randomness,
                'seed' : seed, 'rays' : rays, 'amount' : amount
                }

    def transform(self, image, center, radius, rays, amount, color, randomness, seed):

        def kernel(point, value, center, radius, ray_lengths, amount, color):
            rays = len(ray_lengths)
            dp = point - center
            dist = np.linalg.norm(dp)
            angle = np.arctan2(dp[1], dp[0])
            d = (angle + np.pi) / (2 * np.pi) * rays
            i = int(d)
            f = d - i 

            if radius != 0:
                length = ray_lengths[i % rays] + f * (ray_lengths[(i+1) % rays] - ray_lengths[i % rays])
                g = length**2 / (dist**2 + 1e-4)
                g = g ** ((100 - amount) / 50.0)
                f -= 0.5
                f = 1 - f**2
                f *= g
            f = np.clip(f, 0, 1)
            return color + f * (value - color)

        random_state = np.random.RandomState(seed=seed)
        ray_lengths = [radius + randomness / 100.0 * radius * random_state.randn()\
                for i in range(rays)]

        out = np.array([[kernel(np.array([y,x]), image[y,x,:].astype(np.float32), center, radius, ray_lengths, amount, color)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(out, 0, 255).astype(np.uint8)

class PerlinNoise(Transform):

    name = 'perlin_noise'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        m = np.array([[1,0],[0,1]]) / (32 * self.im_size / 224)
        turbulence = 16.0
        gain = 0.5
        bias = 0.5
        alpha = float_parameter(self.severity, 0.50)
        seed = np.random.randint(low=0, high=2**32)
        return {'m': m, 'turbulence' : turbulence, 'seed': seed,
                'gain': gain, 'bias': bias, 'alpha': alpha}

    def transform(self, image, m, turbulence, seed, gain, bias, alpha):
        
        random_state = np.random.RandomState(seed=seed)
        noise = PerlinNoiseGenerator(random_state)

        def kernel(point, m, turbulence, gain, bias):
            npoint = np.matmul(point, m)
            f = noise.turbulence(npoint[0], npoint[1], turbulence)\
                    if turbulence != 1.0 else noise.noise(npoint[0], npoint[1])
            f = gain * f + bias
            return np.clip(np.array([f,f,f]),0,1.0)

        noise = np.array([[kernel(np.array([y,x]),m,turbulence,gain, bias) for x in range(self.im_size)]\
                for y in range(self.im_size)])
        out = (1 - alpha) * image.astype(np.float32) + 255 * alpha * noise
        return np.clip(out, 0, 255).astype(np.uint8)

class BlueNoise(Transform):

    name = 'blue_noise'
    tags = ['new_corruption']


    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        intensity = float_parameter(self.severity, 196)

        return {'seed' : seed, 'intensity' : intensity}

    def gen_noise(self, random_state):
        center = self.im_size / 2
        power = np.array([[np.linalg.norm(np.array([x,y])-center)\
                for x in range(self.im_size)] for y in range(self.im_size)])

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
        return noise

    def transform(self, image, seed, intensity):
        random_state = np.random.RandomState(seed=seed)
        noise = np.stack([self.gen_noise(random_state) for i in range(3)],axis=2)

        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)

class BrownishNoise(Transform):

    name = 'brownish_noise'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        intensity = float_parameter(self.severity, 64)

        return {'seed' : seed, 'intensity' : intensity}

    def gen_noise(self, random_state):
        center = self.im_size / 2
        power = np.array([[1/(np.linalg.norm(np.array([x,y])-center)**2+1)\
                for x in range(self.im_size)] for y in range(self.im_size)])

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
        return noise

    def transform(self, image, seed, intensity):
        random_state = np.random.RandomState(seed=seed)
        noise = np.stack([self.gen_noise(random_state) for i in range(3)],axis=2)

        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)

class BleachBypass(Transform):

    name = 'bleach_bypass'
    tags = ['new_corruption']

    def sample_parameters(self):
        amount = float_parameter(self.severity, 1.0)
        return { 'amount' : amount }

    def transform(self, image, amount):
        vals = np.array([0.2126, 0.7152, 0.0722]).reshape(1,1,3)
        luma = np.sum(image*vals, axis=2, keepdims=True)/255
        l = np.clip(10.0 * (luma - 0.45), 0, 1.0)
        result1 = 2 * image * luma / 255
        result2 = 1.0 - 2.0 * (1.0 - luma) * (1.0 - image /255)
        out = ((1-l) * result1 + l * result2) * 255

        return ((1-amount) * image + amount * out).astype(np.uint8)

class Technicolor(Transform):

    name = 'technicolor'
    tags = ['new_corruption']

    def sample_parameters(self):
        amount = float_parameter(self.severity, 1.0)
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

class Pseudocolor(Transform):

    name = 'pseudocolor'
    tags = ['new_corruption']

    def sample_parameters(self):
        smoothness = np.random.uniform(low=0.25, high=0.75)
        color0 = np.random.randint(low=0, high=255, size=3).astype(np.uint8)
        color1 = np.random.randint(low=0, high=255, size=3).astype(np.uint8)
        color2 = np.random.randint(low=0, high=255, size=3).astype(np.uint8)
        color3 = np.random.randint(low=0, high=255, size=3).astype(np.uint8)
        color4 = np.random.randint(low=0, high=255, size=3).astype(np.uint8)
        amount = float_parameter(self.severity, 0.5)

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

class HueShift(Transform):

    name = 'hue_shift'
    tags = ['new_corruption']

    def sample_parameters(self):
        amount = float_parameter(self.severity, 0.5)
        if np.random.uniform() < 0.5:
            amount *= -1
        return {'amount' : amount}

    def transform(self, image, amount):
        
        hsv_image = rgb2hsv(image.astype(np.float32)/255)
        hsv_image[:,:,0] += (amount % 1.0)

        return np.clip(hsv2rgb(hsv_image)*255, 0, 255).astype(np.uint8)

class ColorDither(Transform):

    name = 'color_dither'
    tags = ['new_corruption']

    def sample_parameters(self):
        levels = int_parameter(self.severity,10)
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

class ColorBalance(Transform):

    name = 'color_balance'
    tags = ['new_corruption']

    def sample_parameters(self):
        shift = float_parameter(self.severity, 1.0)
        factor = 1.0 + np.random.choice([-1,1]) * shift
        return { 'factor' : factor}

    def transform(self, image, factor):
        enhancer = ImageEnhance.Color(Image.fromarray(image))
        return np.array(enhancer.enhance(factor))


class CheckerBoardCutOut(Transform):

    name = 'checkerboard_cutout'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        angle = np.random.uniform(low=0, high=2*np.pi)
        scales = np.maximum(np.random.uniform(low=0.1, high=0.25) * self.im_size, 1)
        scales = (scales, scales)
        fraction = float_parameter(self.severity, 1.0)
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


class Lines(Transform):

    name = 'lines'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        length = 1.0
        density = float_parameter(self.severity, 1.0)
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


class BlueNoiseSample(Transform):

    name = 'blue_noise_sample'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        threshold = float_parameter(self.severity, 3.0) - 2.5

        return {'seed' : seed, 'threshold' : threshold}

    def transform(self, image, seed, threshold):
        random_state = np.random.RandomState(seed=seed)

        center = self.im_size / 2
        power = np.array([[np.linalg.norm(np.array([x,y])-center)\
                for x in range(self.im_size)] for y in range(self.im_size)])

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

class CausticRefraction(Transform):

    name = 'caustic_refraction'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        time = np.random.uniform(low=0.5, high=2.0)
        size = np.random.uniform(low=0.75, high=1.25) * self.im_size
        #size = self.im_size
        eta = 4.0
        lens_scale = float_parameter(self.severity, 0.5*self.im_size)
        lighting_amount = float_parameter(self.severity, 2.0)
        softening = 1

        return { 'time' : time, 'size' : size, 'eta' : eta, 'lens_scale' : lens_scale, 'lighting_amount': lighting_amount, 'softening' : softening}

    def transform(self, image, time, size, eta, lens_scale, lighting_amount, softening):

        def caustic_noise_kernel(point, time, size):
            point = point / size
            p = (point % 1) * 6.28318530718 - 250

            i = p.copy()
            c = 1.0
            inten = 0.005

            for n in range(5):
                t = time * (1.0 - (3.5 / (n+1)))
                i = p + np.array([np.cos(t-i[0])+np.sin(t+i[1]),np.sin(t-i[1])+np.cos(t+i[0])])
                length = np.sqrt((p[0] / (np.sin(i[0]+t)/inten))**2 + (p[1] / (np.cos(i[1]+t)/inten))**2)
                c += 1.0/length

            c /= 5.0
            c = 1.17 - c ** 1.4
            color = np.clip(np.abs(c) ** 8.0, 0, 1) 
            return np.array([color, color, color])


        def refract(incident, normal, eta):
            if np.abs(np.dot(incident, normal)) >= 1.0 - 1e-3:
                return incident
            angle = np.arccos(np.dot(incident, normal))
            out_angle = np.arcsin(np.sin(angle) / eta)
            out_unrotated = np.array([np.cos(out_angle), np.sin(out_angle), 0.0])
            spectator_dim = np.cross(incident, normal)
            spectator_dim /= np.linalg.norm(spectator_dim)
            orthogonal_dim = np.cross(normal, spectator_dim)
            rotation_matrix = np.stack((normal, orthogonal_dim, spectator_dim), axis=0)
            return np.matmul(np.linalg.inv(rotation_matrix), out_unrotated)

        def luma_at_offset(image, origin, offset):
            pixel_value = image[origin[0]+offset[0], origin[1]+offset[1], :]\
                    if origin[0]+offset[0] >= 0 and origin[0]+offset[0] < image.shape[0]\
                    and origin[1]+offset[1] >= 0 and origin[1]+offset[1] < image.shape[1]\
                    else np.array([0.0,0.0,0])
            return np.dot(pixel_value, np.array([0.2126, 0.7152, 0.0722]))

        def luma_based_refract(point, image, caustics, eta, lens_scale, lighting_amount):
            north_luma = luma_at_offset(caustics, point, np.array([0,-1]))
            south_luma = luma_at_offset(caustics, point, np.array([0, 1]))
            west_luma = luma_at_offset(caustics, point, np.array([-1, 0]))
            east_luma = luma_at_offset(caustics, point, np.array([1,0]))

            lens_normal = np.array([east_luma - west_luma, south_luma - north_luma, 1.0])
            lens_normal = lens_normal / np.linalg.norm(lens_normal)

            refract_vector = refract(np.array([0.0, 0.0, 1.0]), lens_normal, eta) * lens_scale
            refract_vector = np.round(refract_vector, 3)

            out_pixel = bilinear_interpolation(image, point+refract_vector[0:2])
            out_pixel += (north_luma - south_luma) * lighting_amount
            out_pixel += (east_luma - west_luma) * lighting_amount

            return np.clip(out_pixel, 0, 1)

        noise = np.array([[caustic_noise_kernel(np.array([y,x]), time, size)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        noise = gaussian_filter(noise, sigma=softening)

        image = image.astype(np.float32) / 255
        out = np.array([[luma_based_refract(np.array([y,x]), image, noise, eta, lens_scale, lighting_amount)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip((out * 255).astype(np.uint8), 0, 255)

class PinchAndTwirl(Transform):

    name = 'pinch_and_twirl'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        num_per_axis = 5 if self.im_size==224 else 3
        angles = np.array([np.random.choice([1,-1]) * float_parameter(self.severity, np.pi/2) for i in range(num_per_axis ** 2)]).reshape(num_per_axis, num_per_axis)

        amount = float_parameter(self.severity, 0.4) + 0.1
        return {'num_per_axis' : num_per_axis, 'angles' : angles, 'amount' : amount}

    def transform(self, image, num_per_axis, angles, amount):

        def warp_kernel(point, center, radius, amount, angle):
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            dist = np.linalg.norm(point - center)

            if dist > radius or np.round(dist, 3) == 0.0:
                return point

            d = dist / radius
            t = np.sin(np.pi * 0.5 * d) ** (- amount)

            dx *= t
            dy *= t

            e = 1 - d
            a = angle * (e ** 2)
            
            out = center + np.array([dx*np.cos(a) - dy*np.sin(a), dx*np.sin(a) + dy*np.cos(a)])

            return out

        out = image.copy().astype(np.float32)
        grid_size = self.im_size // num_per_axis
        radius = grid_size / 2
        for i in range(num_per_axis):
            for j in range(num_per_axis):
                l, r = i * grid_size, (i+1) * grid_size
                u, d = j * grid_size, (j+1) * grid_size
                center = np.array([u+radius, l+radius])
                out[u:d,l:r,:] = np.array([[bilinear_interpolation(out, warp_kernel(np.array([y,x]), center, radius, amount, angles[i,j]))\
                        for x in np.arange(l,r)] for y in np.arange(u,d)])

        return np.clip(out, 0, 255).astype(np.uint8)

class FishEye(Transform):

    name = 'fish_eye'
    tags = ['new_corruption']

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        density = 0.01 * 224**2 / (self.im_size**2)
        eta = float_parameter(self.severity, 2.0) + 1.0
        radius = max(0.05 * self.im_size, 3)

        return {'seed' : seed, 'density' : density, 'eta': eta, 'radius' : radius}

    def transform(self, image, density, eta, radius, seed):
        
        def warp_kernel(point, center, a, b, eta):
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            x2 = dx**2
            y2 = dy**2
            a2 = a**2
            b2 = b**2
            if (y2 >= (b2 - b2*x2/a2)):
                return point

            r = 1.0 / eta
            z = np.sqrt((1.0 - x2/a2 - y2/b2) * (a*b))
            z2 = z**2

            x_angle = np.arccos(dx / np.sqrt(x2+z2))
            angle_1 = np.pi/2 - x_angle
            angle_2 = np.arcsin(np.sin(angle_1)*r)
            angle_2 = np.pi/2 - x_angle - angle_2
            out_x = point[0] - np.tan(angle_2)*z

            y_angle = np.arccos(dy / np.sqrt(y2+z2))
            angle_1 = np.pi/2 - y_angle
            angle_2 = np.arcsin(np.sin(angle_1)*r)
            angle_2 = np.pi/2 - y_angle - angle_2
            out_y = point[1] - np.tan(angle_2)*z

            return np.array([out_x, out_y])

        random_state = np.random.RandomState(seed=seed)
        num = int(density * self.im_size**2)

        out = image.copy().astype(np.float32)
        for i in range(num):
            center = random_state.uniform(low=0, high=self.im_size, size=2)
            l = max(np.floor(center[1]-radius).astype(np.int), 0)
            r = min(np.ceil(center[1]+radius).astype(np.int), self.im_size)
            u = max(np.floor(center[0]-radius).astype(np.int), 0)
            d = min(np.ceil(center[0]+radius).astype(np.int), self.im_size)
            out[u:d,l:r,:] = np.array([[bilinear_interpolation(out, warp_kernel(np.array([y,x]), center, radius, radius, eta)) for x in np.arange(l,r)] for y in np.arange(u,d)])

        return np.clip(out, 0, 255).astype(np.uint8)


class WaterDrop(Transform):
    
    name = 'water_drop'
    tags = ['new_corruption']

    def sample_parameters(self):
        center = np.array([self.im_size //2, self.im_size//2])
        center = np.random.uniform(low=0.25, high=0.75, size=2) * self.im_size
        radius = self.im_size//2
        amplitude = float_parameter(self.severity, 0.25)
        wavelength = np.random.uniform(low=0.05, high=0.2) * self.im_size
        phase = np.random.uniform(low=0.0, high=2*np.pi)

        return {'center': center, 'radius' : radius, 'amplitude' : amplitude, 'wavelength' : wavelength, 'phase': phase}

    def transform(self, image, center, radius, amplitude, wavelength, phase):

        def warp_kernel(point, center, radius, amplitude, wavelength, phase):

            dx, dy = point - center
            dist = np.linalg.norm(point-center)
            if dist > radius:
                return point

            amount = amplitude * np.sin(dist / wavelength * np.pi * 2 - phase)
            if dist != 0.0:
                amount *= wavelength / dist
            return point + amount * (point - center)


        image = np.array([[bilinear_interpolation(image, warp_kernel(np.array([y,x]), center, radius, amplitude, wavelength, phase))\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(image, 0, 255).astype(np.uint8)

class Ripple(Transform):

    name = 'ripple'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        amplitudes = np.array([float_parameter(self.severity, 0.025)\
                for i in range(2)]) * self.im_size
        wavelengths = np.random.uniform(low=0.1, high=0.3, size=2) * self.im_size
        phases = np.random.uniform(low=0, high=2*np.pi, size=2)
        return {'amplitudes' : amplitudes, 'wavelengths' : wavelengths, 'phases' : phases}

    def transform(self, image, wavelengths, phases, amplitudes):

        def warp_kernel(point, wavelengths, phases, amplitudes):
            return point + amplitudes * np.sin(2 * np.pi * point / wavelengths + phases)

        image = np.array([[bilinear_interpolation(image, warp_kernel(np.array([y,x]), wavelengths, phases, amplitudes))\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(image, 0, 255).astype(np.uint8) 


class PerspectiveNoBars(Transform):

    name = 'perspective_no_bars'
    tags = ['new_corruption']

    def sample_parameters(self):
        offset_x = float_parameter(self.severity, 0.1)
        if np.random.uniform() > 0.5:
            offset_x = -offset_x
        offset_y = float_parameter(self.severity, 0.1)
        if np.random.uniform() > 0.5:
            offset_y = -offset_y
        shift_x = float_parameter(self.severity, self.im_size / 10)
        if np.random.uniform() > 0.5:
            shift_x = -shift_x
        shift_y = float_parameter(self.severity, self.im_size / 10)
        if np.random.uniform() > 0.5:
            shift_y = -shift_y
        factor_x = float_parameter(self.severity, 0.15)
        if np.random.uniform() > 0.5:
            factor_x = -factor_x
        factor_x = 2 ** factor_x
        factor_y = float_parameter(self.severity, 0.15)
        if np.random.uniform() > 0.5:
            factor_y = -factor_y
        factor_y = 2 ** factor_y
        denom_x = float_parameter(self.severity, 0.2 / self.im_size)
        if np.random.uniform() > 0.5:
            denom_x = denom_x
        denom_y = float_parameter(self.severity, 0.2 / self.im_size)
        if np.random.uniform() > 0.5:
            denom_y = denom_y
        perspective_params = np.array([factor_x, offset_x, shift_x,offset_y, factor_y, shift_y, denom_x, denom_y])
        return {'perspective_params' : perspective_params}

    def transform(self, image, perspective_params):
        im = Image.fromarray(image)
        im = im.transform(
                (self.im_size, self.im_size), 
                Image.PERSPECTIVE,
                perspective_params, 
                resample=Image.BILINEAR
                )
        im = np.array(im).astype(np.float32)
        mask = Image.fromarray(np.ones_like(image).astype(np.uint8)*255)
        mask = mask.transform(
                (self.im_size, self.im_size),
                Image.PERSPECTIVE,
                perspective_params,
                resample=Image.BILINEAR
                )
        mask = np.array(mask).astype(np.float32) / 255
        im = mask * im + (1-mask) * image

        return im.astype(np.uint8)

    def convert_to_numpy(self, params):
        return params['perspective_params']

    def convert_from_numpy(self, numpy_record):
        return {'perspective_params' : numpy_record}


class QuadrilateralNoBars(Transform):

    name = 'quadrilateral_no_bars'
    tags = ['new_corruption']

    def sample_parameters(self):
        points = np.array([
            [0,0],
            [0, self.im_size],
            [self.im_size, self.im_size],
            [self.im_size, 0]
            ]).astype(np.float32)
        shift = float_parameter(self.severity, self.im_size / 3) * np.random.uniform(low=-1,high=1, size=(4,2))
        points += shift
        return {'points' : points}

    def transform(self, image, points):
        im = Image.fromarray(image)
        im = im.transform(
                (self.im_size, self.im_size), 
                Image.QUAD, 
                points.flatten(), 
                resample=Image.BILINEAR
                )
        im = np.array(im).astype(np.float32)
        mask = Image.fromarray(np.ones_like(image).astype(np.uint8)*255)
        mask = mask.transform(
                (self.im_size, self.im_size),
                Image.QUAD,
                points.flatten(),
                resample=Image.BILINEAR
                )
        mask = np.array(mask).astype(np.float32) / 255
        im = mask * im + (1-mask) * image

        return im.astype(np.uint8)

    def convert_to_numpy(self, params):
        return params['points'].flatten()

    def convert_from_numpy(self, numpy_record):
        return {'points' : numpy_record.reshape(4,2)}

class Scatter(Transform):

    name = 'scatter'
    tags = ['new_corruption']

    def sample_parameters(self):
        seed = np.random.uniform(low=0.0, high=10.0)
        radius = float_parameter(self.severity,  self.im_size/10)

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

class ChromaticAbberation(Transform):

    name = 'chromatic_abberation'
    tags = ['new_corruption']

    def sample_parameters(self):
        angles = np.random.uniform(low=0, high=2*np.pi, size=3)
        dists = np.array([float_parameter(self.severity, self.im_size / 10)\
                for i in range(3)])
        shifts = np.array([[np.cos(a)*d, np.sin(a)*d] for a, d in zip(angles, dists)])
        return { 'rgb_shifts' : shifts}

    def transform(self, image, rgb_shifts):
        out = image.copy()
        for i in range(3):
            out[:,:,i] = shift(image[:,:,i], rgb_shifts[i], prefilter=False)
        return out

    def convert_to_numpy(self, params):
        return params['rgb_shifts'].flatten()

    def convert_from_numpy(self, numpy_record):
        return {'rgb_shifts' : numpy_record.reshape(3,2).astype(np.int)}

class TransverseChromaticAbberation(Transform):

    name = 'transverse_chromatic_abberation'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        scales = np.array([float_parameter(self.severity, 0.5)\
                for i in range(3)])
        scale = float_parameter(self.severity, 0.5)
        scales = np.array([1.0, 1.0+scale/2, 1.0+scale])
        scales = scales[np.random.permutation(3)]

        return { 'scales' : scales }

    def transform(self, image, scales):
        out = image.copy()
        for c in range(3):
            zoomed = zoom(image[:,:,c], scales[c], prefilter=False)
            edge = (zoomed.shape[0]-self.im_size)//2
            out[:,:,c] = zoomed[edge:edge+self.im_size, edge:edge+self.im_size]
        return out.astype(np.uint8)
            
    def convert_to_numpy(self, params):
        return params['scales'].flatten()

    def convert_from_numpy(self, numpy_record):
        return {'scales' : numpy_record}


class CircularMotionBlur(Transform):

    name = 'circular_motion_blur'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        amount = float_parameter(self.severity,15)

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
        
