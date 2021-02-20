# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Augmentation
from .utils.severity import float_parameter, int_parameter, sample_level
from .utils.image import smoothstep
from .utils.noise import PerlinNoiseGenerator
from scipy.fftpack import ifft2
import numpy as np

class SingleFrequencyGreyscale(Augmentation):

    tags = ['additive_noise', 'single_frequency_greyscale']
    name = 'single_frequency_greyscale'

    def sample_parameters(self):
        freq_mag = np.random.uniform(low=-np.pi, high=np.pi)
        freq_2 = np.random.uniform(low=-abs(freq_mag), high=abs(freq_mag))
        freq = np.array([freq_mag, freq_2])[np.random.permutation(2)]
        phase = np.random.uniform(low=0, high=2*np.pi)
        intensity = float_parameter(sample_level(self.severity,self.max_intensity), 196)
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

class SingleFrequencyColor(Augmentation):

    tags = ['additive_noise', 'color', 'single_frequency_color']
    name = 'single_frequency_color'

    def sample_parameters(self):
        freq = np.random.uniform(low=0, high=np.pi, size=2)
        phase = np.random.uniform(low=0, high=2*np.pi)
        intensity = [float_parameter(sample_level(self.severity,self.max_intensity), 196) for i in range(3)]
        return { 'freq' : freq, 'phase' : phase, 'intensity' : intensity}

    def transform(self, image, freq, phase, intensity):
        noise = np.array([[np.sin(x * freq[0] + y * freq[1] + phase)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        noise = np.stack((intensity[0] * noise, intensity[1] * noise, intensity[2] * noise), axis=2)

        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def convert_to_numpy(self, params):
        return np.array(params['freq'].tolist() + [params['phase']] + params['intensity'])

    def convert_from_numpy(self, numpy_record):
        return {'freq' : numpy_record[0:2],
                'phase' : numpy_record[2],
                'intensity' : numpy_record[3:6].tolist()
                }

class CocentricSineWaves(Augmentation):

    tags = ['additive_noise', 'filterpedia', 'color', 'cocentric_sine_waves']
    name = 'cocentric_sine_waves'

    def sample_parameters(self):
        offset = np.random.uniform(low=0, high=self.im_size, size=2)
        freq = np.random.uniform(low=0, high=10)
        amplitude = np.random.uniform(low=0, high=self.im_size/10)
        ring_width = np.random.uniform(low=0, high=self.im_size/10)
        intensity = [float_parameter(sample_level(self.severity,self.max_intensity), 128) for i in range(3)]

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
        

class PlasmaNoise(Augmentation):

    tags = ['additive_noise', 'color', 'filterpedia', 'plasma_noise', 'slow']
    name = 'plasma_noise'

    def sample_parameters(self):
        time = np.random.uniform(low=0.0, high=6*np.pi)
        iterations = np.random.randint(low=4, high=7)
        sharpness = np.random.uniform(low=0.5, high=1.0)
        scale = np.random.uniform(low=0.075, high=0.2) * self.im_size
        intensity = float_parameter(sample_level(self.severity,self.max_intensity),64)
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


class VoronoiNoise(Augmentation):

    tags = ['additive_noise', 'filterpedia', 'voronoi_noise', 'slow']
    name = 'voronoi_noise'

    def sample_parameters(self):
        seed = np.random.uniform(low=0, high=10)
        density = np.random.uniform(low=0.5, high=0.9)
        size = np.random.uniform(low=0.05, high=0.2) * self.im_size
        intensity = float_parameter(sample_level(self.severity,self.max_intensity),255)
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

class CausticNoise(Augmentation):

    tags = ['additive_noise', 'filterpedia']
    name = 'caustic_noise'

    def sample_parameters(self):
        time = np.random.uniform(low=0.5, high=2.0)
        size = np.random.uniform(low=0.75, high=1.25) * self.im_size
        #size = self.im_size
        intensity = float_parameter(sample_level(self.severity,self.max_intensity), 255)

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
        #return np.clip(255 *  noise, 0, 255).astype(np.uint8)

class Sparkles(Augmentation):

    tags = ['additive_noise']
    name = 'sparkles'

    def sample_parameters(self):
        centers = np.random.uniform(low=0, high=self.im_size, size=(5, 2))
        radii = np.array([float_parameter(sample_level(self.severity, self.max_intensity), 0.1)\
                for i in range(5)]) * self.im_size
        #radii = np.array([0.1 for i in range(5)]) * self.im_size
        #amounts = np.array([float_parameter(sample_level(self.severity, self.max_intensity), 50)\
        #        for i in range(5)])
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


class InverseSparkles(Augmentation):

    tags = ['additive_noise']
    name = 'inverse_sparkles'

    def sample_parameters(self):
        center = np.random.uniform(low=0.25, high=0.75, size=2) * self.im_size
        #radius = self.im_size // 4
        #radius = float_parameter(sample_level(self.severity, self.max_intensity), 0.5)
        #radius = (0.75 - radius) * self.im_size
        radius = 0.25 * self.im_size
        #amount = 25
        amount = 100
        amount = float_parameter(sample_level(self.severity, self.max_intensity), 65)
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

class PerlinNoise(Augmentation):

    tags = ['additive_noise']
    name = 'perlin_noise'

    def sample_parameters(self):
        m = np.array([[1,0],[0,1]]) / (32 * self.im_size / 224)
        turbulence = 16.0
        gain = 0.5
        bias = 0.5
        alpha = float_parameter(sample_level(self.severity, self.im_size), 0.50)
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

class BlueNoise(Augmentation):

    tags = ['additive_noise']
    name = 'blue_noise'


    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        intensity = float_parameter(sample_level(self.severity, self.max_intensity), 196)

        return {'seed' : seed, 'intensity' : intensity}

    def gen_noise(self, random_state):
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
        return noise

    def transform(self, image, seed, intensity):
        random_state = np.random.RandomState(seed=seed)
        noise = np.stack([self.gen_noise(random_state) for i in range(3)],axis=2)
        #luma_noise = noise.reshape(self.im_size, self.im_size, 1) * np.array([[[0.2126, 0.7152, 0.0722]]])

        #return np.clip(image + intensity * luma_noise, 0, 255).astype(np.uint8)
        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)

class BrownishNoise(Augmentation):

    tags = ['additive_noise']
    name = 'brownish_noise'

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        intensity = float_parameter(sample_level(self.severity, self.max_intensity), 64)

        return {'seed' : seed, 'intensity' : intensity}

    def gen_noise(self, random_state):
        center = self.im_size / 2
        power = np.array([[1/(np.linalg.norm(np.array([x,y])-center)**2+1)\
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
        return noise

    def transform(self, image, seed, intensity):
        random_state = np.random.RandomState(seed=seed)
        noise = np.stack([self.gen_noise(random_state) for i in range(3)],axis=2)
        #luma_noise = noise.reshape(self.im_size, self.im_size, 1) * np.array([[[0.2126, 0.7152, 0.0722]]])

        #return np.clip(image + intensity * luma_noise, 0, 255).astype(np.uint8)
        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)
