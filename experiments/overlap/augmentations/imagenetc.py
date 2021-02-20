# This source code is adapted from code licensed under the license at
# third_party/imagenetc_license from the root directory of the repository
# Originally available: github.com/hendrycks/robustness

# Modifications Copyright (c) Facebook, Inc. and its affiliates, 
# licensed under the MIT license found in the LICENSE file in the root 
# directory of this source tree.


from .base import Augmentation


import pickle
import torch
import torch.utils.data

# Per-channel mean and SD values in BGR order
_MEAN = [125.3, 123.0, 113.9]
_SD = [63.0, 62.1, 66.7]

import os
from PIL import Image
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np

from PIL import Image


# /////////////// Distortion Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

warnings.simplefilter("ignore", UserWarning)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(seed, mapsize, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100
    random_state = np.random.RandomState(seed=seed)

    def wibbledmean(array):
        return array / 4 + wibble * random_state.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////

def gaussian_noise(x, im_size, seed, severity=1):
    if im_size == 32:
        c = [0.04, 0.06, .08, .09, .10][int(severity) - 1]
    else:
        c = [.08, .12, 0.18, 0.26, 0.38][int(severity) - 1]
    random_state = np.random.RandomState(seed=seed)

    x = np.array(x) / 255.
    return np.clip(x + random_state.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, im_size, seed, severity=1):
    if im_size == 32:
        c = [500, 250, 100, 75, 50][int(severity) - 1]
    else:
        c = [60, 25, 12, 5, 3][int(severity) - 1]

    random_state = np.random.RandomState(seed=seed)

    x = np.array(x) / 255.
    return np.clip(random_state.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, im_size, seed, severity=1):
    if im_size == 32:
        c = [.01, .02, .03, .05, .07][int(severity) - 1]
    else:
        c = [.03, .06, .09, 0.17, 0.27][int(severity) - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c, seed=seed)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, im_size, seed, severity=1):
    if im_size == 32:
        c = [.06, .1, .12, .16, .2][int(severity) - 1]
    else:
        c = [.15, .2, 0.35, 0.45, 0.6][int(severity) - 1]

    random_state = np.random.RandomState(seed=seed)

    x = np.array(x) / 255.
    return np.clip(x + x * random_state.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, im_size, severity=1):
    if im_size == 32:
        c = [.4, .6, 0.7, .8, 1][int(severity) - 1]
    else:
        c = [1, 2, 3, 4, 6][int(severity) - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, im_size, seed, severity=1):
    # sigma, max_delta, iterations
    if im_size == 32:
        c = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)][int(severity) - 1]
    else:
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][int(severity) - 1]

    random_state = np.random.RandomState(seed=seed)

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(im_size - c[1], c[1], -1):
            for w in range(im_size - c[1], c[1], -1):
                dx, dy = random_state.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(x, im_size, severity=1):
    if im_size == 32:
        c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][int(severity) - 1]
    else:
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][int(severity) - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, im_size, angle, severity=1):
    if im_size == 32:
        c = [(6,1), (6,1.5), (6,2), (8,2), (9,2.5)][int(severity) - 1]
    else:
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][int(severity) - 1]


    output = BytesIO()
    x = Image.fromarray(x)
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=angle)

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if x.shape != (im_size, im_size):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, im_size, severity=1):
    if im_size == 32:
        c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
            np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][int(severity) - 1]
    else:
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][int(severity) - 1]


    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, im_size, seed, severity=1):
    if im_size == 32:
        c = [(.2,3), (.5,3), (0.75,2.5), (1,2), (1.5,1.75)][int(severity) - 1]
        mapsize = 32
    else:
        c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][int(severity) - 1]
        mapsize = 256


    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1], seed=seed, mapsize=mapsize)[:im_size, :im_size][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, im_size, frost_path, image_idx, crop_pos, severity=1):
    if im_size == 32:
        c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][int(severity) - 1]
    else:
        c = [(1, 0.4),
             (0.8, 0.6),
             (0.7, 0.7),
             (0.65, 0.7),
             (0.6, 0.75)][int(severity) - 1]

    idx = image_idx
    filename = ['./frost1.png', './frost2.png', './frost3.png', './frost4.jpg', './frost5.jpg', './frost6.jpg'][idx]
    filename = os.path.join(frost_path, filename)
    frost = cv2.imread(filename)
    if im_size == 32:
        frost = cv2.resize(frost, (0, 0), fx=0.2, fy=0.2)
    # randomly crop and convert to rgb
    #x_start, y_start = np.random.randint(0, frost.shape[0] - 32), np.random.randint(0, frost.shape[1] - 32)
    x_start, y_start = crop_pos[0], crop_pos[1]
    frost = frost[x_start:x_start + im_size, y_start:y_start + im_size][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, im_size, seed, severity=1):
    if im_size == 32:
        c = [(0.1,0.2,1,0.6,8,3,0.95),
             (0.1,0.2,1,0.5,10,4,0.9),
             (0.15,0.3,1.75,0.55,10,4,0.9),
             (0.25,0.3,2.25,0.6,12,6,0.85),
             (0.3,0.3,1.25,0.65,14,12,0.8)][int(severity) - 1]
    else:
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
             (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][int(severity) - 1]

    random_state = np.random.RandomState(seed=seed)

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = random_state.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=random_state.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(im_size, im_size, 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, im_size, seed, severity=1):
    if im_size == 32:
        c = [(0.62,0.1,0.7,0.7,0.5,0),
             (0.65,0.1,0.8,0.7,0.5,0),
             (0.65,0.3,1,0.69,0.5,0),
             (0.65,0.1,0.7,0.69,0.6,1),
             (0.65,0.1,0.5,0.68,0.6,1)][int(severity) - 1]
    else:
        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
             (0.65, 0.3, 3, 0.68, 0.6, 0),
             (0.65, 0.3, 2, 0.68, 0.5, 0),
             (0.65, 0.3, 1, 0.65, 1.5, 1),
             (0.67, 0.4, 1, 0.65, 1.5, 1)][int(severity) - 1]

    x = np.array(x, dtype=np.float32) / 255.
    random_state = np.random.RandomState(seed=seed)

    liquid_layer = random_state.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, im_size, severity=1):
    if im_size == 32:
        c = [.75, .5, .4, .3, 0.15][int(severity) - 1]
    else:
        c = [0.4, .3, .2, .1, .05][int(severity) - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, im_size, severity=1):
    if im_size == 32:
        c = [.05, .1, .15, .2, .3][int(severity) - 1]
    else:
        c = [.1, .2, .3, .4, .5][int(severity) - 1]


    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, im_size, severity=1):
    if im_size == 32:
        c = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][int(severity) - 1]
    else:
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][int(severity) - 1]


    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, im_size, severity=1):
    if im_size == 32:
        c = [80, 65, 58, 50, 40][int(severity) - 1]
    else:
        c = [25, 18, 15, 10, 7][int(severity) - 1]
    x = Image.fromarray(x)

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, im_size, severity=1):
    if im_size == 32:
        c = [0.95, 0.9, 0.85, 0.75, 0.65][int(severity) - 1]
    else:
        c = [0.6, 0.5, 0.4, 0.3, 0.25][int(severity) - 1]

    x = Image.fromarray(x)

    x = x.resize((int(im_size * c), int(im_size * c)), PILImage.BOX)
    x = x.resize((im_size, im_size), PILImage.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, im_size, seed, severity=1):
    IMSIZE = im_size
    if im_size == 32:
        c = [(IMSIZE*0, IMSIZE*0, IMSIZE*0.08),
            (IMSIZE*0.05, IMSIZE*0.2, IMSIZE*0.07),
            (IMSIZE*0.08, IMSIZE*0.06, IMSIZE*0.06),
            (IMSIZE*0.1, IMSIZE*0.04, IMSIZE*0.05),
            (IMSIZE*0.1, IMSIZE*0.03, IMSIZE*0.03)][int(severity) - 1]
    else:
        c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
             (244 * 2, 244 * 0.08, 244 * 0.2),
             (244 * 0.05, 244 * 0.01, 244 * 0.02),
             (244 * 0.07, 244 * 0.01, 244 * 0.02),
             (244 * 0.12, 244 * 0.01, 244 * 0.02)][int(severity) - 1]
    random_state = np.random.RandomState(seed=seed)

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(random_state.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(random_state.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255

class GaussianNoise(Augmentation):

    tags = ['imagenet_c', 'noise']
    name = 'gaussian_noise'

    def sample_parameters(self):
        seed = np.random.randint(0,2**32)
        return {'seed': seed}

    def transform(self, image, seed):
        return np.uint8(gaussian_noise(image, self.im_size, seed, severity=self.severity))

class ShotNoise(Augmentation):

    tags = ['imagenet_c', 'noise']
    name = 'shot_noise'

    def sample_parameters(self):
        seed = np.random.randint(0,2**32)
        return {'seed': seed}

    def transform(self, image, seed):
        return np.uint8(shot_noise(image, self.im_size, seed, severity=self.severity))

class ImpulseNoise(Augmentation):

    tags = ['imagenet_c', 'noise']
    name = 'impulse_noise'

    def sample_parameters(self):
        seed = np.random.randint(0,2**32)
        return {'seed': seed}

    def transform(self, image, seed):
        return np.uint8(impulse_noise(image, self.im_size, seed, severity=self.severity))

class SpeckleNoise(Augmentation):

    tags = ['imagenet_c', 'extra']
    name = 'speckle_noise'

    def sample_parameters(self):
        seed = np.random.randint(0,2**32)
        return {'seed': seed}

    def transform(self, image,  seed):
        return np.uint8(speckle_noise(image, self.im_size, seed, severity=self.severity))

class ElasticTransform(Augmentation):

    tags = ['imagenet_c', 'digital']
    name = 'elastic_transform'

    def sample_parameters(self):
        seed = np.random.randint(0,2**32)
        return {'seed': seed}

    def transform(self, image,  seed):
        return np.uint8(elastic_transform(image, self.im_size, seed, severity=self.severity))

class GlassBlur(Augmentation):

    tags = ['imagenet_c', 'blur']
    name = 'glass_blur'

    def sample_parameters(self):
        seed = np.random.randint(0,2**32)
        return {'seed': seed}

    def transform(self, image,  seed):
        return np.uint8(glass_blur(image, self.im_size, seed, severity=self.severity))

class Snow(Augmentation):

    tags = ['imagenet_c', 'weather']
    name = 'snow'

    def sample_parameters(self):
        seed = np.random.randint(0,2**32)
        return {'seed': seed}

    def transform(self, image,  seed):
        return np.uint8(snow(image, self.im_size, seed, severity=self.severity))

class Spatter(Augmentation):

    tags = ['imagenet_c', 'extra']
    name = 'spatter'

    def sample_parameters(self):
        seed = np.random.randint(0,2**32)
        return {'seed': seed}

    def transform(self, image,  seed):
        return np.uint8(spatter(image, self.im_size, seed, severity=self.severity))

class Fog(Augmentation):

    tags = ['imagenet_c', 'blur']
    name = 'fog'

    def sample_parameters(self):
        seed = np.random.randint(0,2**32)
        return {'seed': seed}

    def transform(self, image,  seed):
        return np.uint8(fog(image, self.im_size, seed, severity=self.severity))

class ZoomBlur(Augmentation):

    tags = ['imagenet_c', 'blur']
    name = 'zoom_blur'

    def sample_parameters(self):
        return {}

    def transform(self, image):
        return np.uint8(zoom_blur(image, self.im_size, severity=self.severity))

class Pixelate(Augmentation):

    tags = ['imagenet_c', 'digital']
    name = 'pixelate'

    def sample_parameters(self):
        return {}

    def transform(self, image):
        return np.uint8(pixelate(image, self.im_size, severity=self.severity))

class JPEGCompression(Augmentation):

    tags = ['imagenet_c', 'digital']
    name = 'jpeg_compression'

    def sample_parameters(self):
        return {}

    def transform(self, image):
        return np.uint8(jpeg_compression(image, self.im_size, severity=self.severity))

class Contrast(Augmentation):

    tags = ['imagenet_c', 'digital']
    name = 'contrast'

    def sample_parameters(self):
        return {}

    def transform(self, image):
        return np.uint8(contrast(image, self.im_size, severity=self.severity))

class Brightness(Augmentation):

    tags = ['imagenet_c', 'weather']
    name = 'brightness'

    def sample_parameters(self):
        return {}

    def transform(self, image):
        return np.uint8(brightness(image, self.im_size, severity=self.severity))

class MotionBlur(Augmentation):

    tags = ['imagenet_c', 'blur']
    name = 'motion_blur'

    def sample_parameters(self):
        angle = np.random.uniform(-45,45)
        return {'angle' : angle}

    def transform(self, image,  angle):
        return np.uint8(motion_blur(image, self.im_size, angle=angle, severity=self.severity))

class GaussianBlur(Augmentation):

    tags = ['imagenet_c', 'extra']
    name = 'gaussian_blur'

    def sample_parameters(self):
        return {}

    def transform(self, image):
        return np.uint8(gaussian_blur(image, self.im_size, severity=self.severity))

class Frost(Augmentation):

    tags = ['imagenet_c', 'path_required', 'weather']
    name = 'frost'

    def __init__(self, severity, im_size, record=False, max_intensity=False, frost_path=None):
        super().__init__(severity, im_size, record, max_intensity)
        self.frost_path = frost_path


    def sample_parameters(self):
        image_idx = np.random.randint(5)
        filename = ['./frost1.png', './frost2.png', './frost3.png', './frost4.jpg', './frost5.jpg', './frost6.jpg'][image_idx]
        filename = os.path.join(self.frost_path, filename)
        frost = cv2.imread(filename)
        if self.im_size == 32:
            frost = cv2.resize(frost, (0, 0), fx=0.2, fy=0.2)
        x_start, y_start = np.random.randint(0, frost.shape[0] - self.im_size), np.random.randint(0, frost.shape[1] - self.im_size)

        return {'image_idx' : image_idx, 'crop_pos' : (x_start, y_start)}

    def transform(self, image, image_idx, crop_pos):
        return np.uint8(frost(image, self.im_size, frost_path=self.frost_path, image_idx=image_idx, crop_pos=crop_pos, severity=self.severity))

    def convert_to_numpy(self, params):
        return np.array([params['image_idx']] + list( params['crop_pos']))

    def convert_from_numpy(self, numpy_record):
        return {'image_idx' : int(numpy_record[0]), 'crop_pos' : tuple(numpy_record[1:].astype(np.int).tolist())}

class DefocusBlur(Augmentation):

    tags = ['imagenet_c', 'blur']
    name = 'defocus_blur'

    def sample_parameters(self):
        return {}

    def transform(self, image):
        return np.uint8(defocus_blur(image, self.im_size, severity=self.severity))

class Saturate(Augmentation):

    tags = ['imagenet_c', 'extra']
    name = 'saturate'

    def sample_parameters(self):
        return {}

    def transform(self, image):
        return np.uint8(saturate(image, self.im_size, severity=self.severity))
