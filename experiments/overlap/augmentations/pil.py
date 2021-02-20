# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Augmentation
from .utils.severity import float_parameter, int_parameter, sample_level

from PIL import Image, ImageOps, ImageEnhance
import numpy as np

class AutoContrast(Augmentation):

    tags = ['autoaugment', 'augmix', 'pil', 'color', 'autocontrast']
    name = 'autocontrast'

    def sample_parameters(self):
        return {}

    def transform(self, image):
        im = ImageOps.autocontrast(Image.fromarray(image))
        return np.array(im)


class Equalize(Augmentation):

    tags = ['autoaugment', 'augmix', 'pil', 'color', 'equalize']
    name = 'equalize'

    def sample_parameters(self):
        return {}

    def transform(self, image):
        im = ImageOps.equalize(Image.fromarray(image))
        return np.array(im)

class Posterize(Augmentation):

    tags = ['autoaugment', 'augmix', 'pil', 'color', 'posterize']
    name = 'posterize'

    def sample_parameters(self):
        bits = 4 - int_parameter(sample_level(self.severity,self.max_intensity), 4)
        return {'bits' : bits}

    def transform(self, image, bits):
        im = ImageOps.posterize(Image.fromarray(image), int(bits))
        return np.array(im)

class Solarize(Augmentation):

    tags = ['autoaugment', 'augmix', 'pil', 'color', 'solarize']
    name = 'solarize'

    def sample_parameters(self):
        threshold = 256 - int_parameter(sample_level(self.severity,self.max_intensity), 256)
        return {'threshold' : threshold}

    def transform(self, image, threshold):
        im = ImageOps.solarize(Image.fromarray(image), threshold)
        return np.array(im)

class Affine(Augmentation):

    tags = ['pil', 'spatial', 'affine']
    name = 'affine'

    def sample_parameters(self):
        offset_x = float_parameter(sample_level(self.severity,self.max_intensity), 0.3)
        if np.random.uniform() > 0.5:
            offset_x = -offset_x
        offset_y = float_parameter(sample_level(self.severity,self.max_intensity), 0.3)
        if np.random.uniform() > 0.5:
            offset_y = -offset_y
        shift_x = float_parameter(sample_level(self.severity,self.max_intensity), self.im_size / 3)
        if np.random.uniform() > 0.5:
            shift_x = -shift_x
        shift_y = float_parameter(sample_level(self.severity,self.max_intensity), self.im_size / 3)
        if np.random.uniform() > 0.5:
            shift_y = -shift_y
        factor_x = float_parameter(sample_level(self.severity,self.max_intensity), 0.5)
        if np.random.uniform() > 0.5:
            factor_x = -factor_x
        factor_x = 2 ** factor_x
        factor_y = float_parameter(sample_level(self.severity,self.max_intensity), 0.5)
        if np.random.uniform() > 0.5:
            factor_y = -factor_y
        factor_y = 2 ** factor_y
        affine_matrix = np.array([[factor_x, offset_x, shift_x],[offset_y, factor_y, shift_y]])
        return {'affine_matrix' : affine_matrix}

    def transform(self, image, affine_matrix):
        im = Image.fromarray(image)
        im = im.transform(
                (self.im_size, self.im_size), 
                Image.AFFINE, 
                affine_matrix.flatten(), 
                resample=Image.BILINEAR
                )
        return np.array(im)

    def convert_to_numpy(self, params):
        return params['affine_matrix'].flatten()

    def convert_from_numpy(self, numpy_record):
        return {'affine_matrix' : numpy_record.reshape(2,3)}

class ShearX(Affine):

    tags = ['autoaugment', 'augmix', 'pil', 'spatial', 'shear_x']
    name = 'shear_x'

    def sample_parameters(self):
        offset = float_parameter(sample_level(self.severity,self.max_intensity), 0.3)
        if np.random.uniform() > 0.5:
            offset = -offset
        return {'offset' : offset}

    def transform(self, image, offset):
        affine_matrix = np.array([[1, offset, 0],[0, 1, 0]])
        return super().transform(image, affine_matrix)

    def convert_to_numpy(self, params):
        return np.array([params['offset']])

    def convert_from_numpy(self, numpy_record):
        return {'offset' : numpy_record[0]}


class ShearY(Affine):

    tags = ['autoaugment', 'augmix', 'pil', 'spatial', 'shear_y']
    name = 'shear_y'

    def sample_parameters(self):
        offset = float_parameter(sample_level(self.severity,self.max_intensity), 0.3)
        if np.random.uniform() > 0.5:
            offset = -offset
        return {'offset' : offset}

    def transform(self, image, offset):
        affine_matrix = np.array([[1, 0, 0],[offset, 1, 0]])
        return super().transform(image, affine_matrix)

    def convert_to_numpy(self, params):
        return np.array([params['offset']])

    def convert_from_numpy(self, numpy_record):
        return {'offset' : numpy_record[0]}


class TranslateX(Affine):

    tags = ['autoaugment', 'augmix', 'pil', 'spatial', 'translate_x']
    name = 'translate_x'

    def sample_parameters(self):
        offset = float_parameter(sample_level(self.severity,self.max_intensity), self.im_size / 3)
        if np.random.uniform() > 0.5:
            offset = -offset
        return {'offset' : offset}

    def transform(self, image, offset):
        affine_matrix = np.array([[1, 0, offset],[0, 1, 0]])
        return super().transform(image, affine_matrix)

    def convert_to_numpy(self, params):
        return np.array([params['offset']])

    def convert_from_numpy(self, numpy_record):
        return {'offset' : numpy_record[0]}

class TranslateY(Affine):

    tags = ['autoaugment', 'augmix', 'pil', 'spatial', 'translate_y']
    name = 'translate_y'

    def sample_parameters(self):
        offset = float_parameter(sample_level(self.severity,self.max_intensity), self.im_size / 3)
        if np.random.uniform() > 0.5:
            offset = -offset
        return {'offset' : offset}

    def transform(self, image, offset):
        affine_matrix = np.array([[1, 0, 0],[0, 1, offset]])
        return super().transform(image, affine_matrix)

    def convert_to_numpy(self, params):
        return np.array([params['offset']])

    def convert_from_numpy(self, numpy_record):
        return {'offset' : numpy_record[0]}

class Rotate(Augmentation):

    tags = ['autoaugment', 'augmix', 'pil', 'spatial', 'rotate']
    name = 'rotate'

    def sample_parameters(self):
        degrees = float_parameter(sample_level(self.severity,self.max_intensity), 30)
        if np.random.uniform() > 0.5:
            degrees = -degrees
        return {'degrees' : degrees}

    def transform(self, image, degrees):
        im = Image.fromarray(image)
        im = im.rotate(degrees, resample=Image.BILINEAR)
        return np.array(im)

class Invert(Augmentation):

    tags = ['autoaugment', 'pil', 'color', 'invert']
    name = 'invert'

    def sample_parameters(self):
        return {}

    def transform(self, image):
        im = ImageOps.invert(Image.fromarray(image))
        return np.array(im)


class ColorBalance(Augmentation):

    tags = ['autoaugment', 'pil', 'color', 'color_balance']
    name = 'color_balance'

    def sample_parameters(self):
        shift = float_parameter(sample_level(self.severity, self.max_intensity), 1.0)
        factor = 1.0 + np.random.choice([-1,1]) * shift
        return { 'factor' : factor}

    def transform(self, image, factor):
        enhancer = ImageEnhance.Color(Image.fromarray(image))
        return np.array(enhancer.enhance(factor))


class Sharpness(Augmentation):

    tags = ['autoaugment', 'pil', 'blur', 'sharpness']
    name = 'sharpness'

    def sample_parameters(self):
        shift = float_parameter(sample_level(self.severity, self.max_intensity), 1.0)
        factor = 1.0 + np.random.choice([-1,1]) * shift
        return { 'factor' : factor}

    def transform(self, image, factor):
        enhancer = ImageEnhance.Sharpness(Image.fromarray(image))
        return np.array(enhancer.enhance(factor))

class Contrast(Augmentation):

    tags = ['autoaugment', 'pil', 'color', 'imagenet_c_overlap', 'contrast']
    name = 'contrast_pil'

    def sample_parameters(self):
        shift = float_parameter(sample_level(self.severity, self.max_intensity), 1.0)
        factor = 1.0 + np.random.choice([-1,1]) * shift
        return { 'factor' : factor}

    def transform(self, image, factor):
        enhancer = ImageEnhance.Contrast(Image.fromarray(image))
        return np.array(enhancer.enhance(factor))

class Brightness(Augmentation):

    tags = ['autoaugment', 'pil', 'color', 'imagenet_c_overlap', 'brightness']
    name = 'brightness_pil'

    def sample_parameters(self):
        shift = float_parameter(sample_level(self.severity, self.max_intensity), 1.0)
        factor = 1.0 + np.random.choice([-1,1]) * shift
        return { 'factor' : factor}

    def transform(self, image, factor):
        enhancer = ImageEnhance.Brightness(Image.fromarray(image))
        return np.array(enhancer.enhance(factor))


class ScaleX(Affine):

    tags = ['pil', 'spatial', 'scale_x']
    name = 'scale_x'

    def sample_parameters(self):
        factor = float_parameter(sample_level(self.severity,self.max_intensity), 0.5)
        if np.random.uniform() > 0.5:
            factor = -factor
        factor = 2 ** factor
        return {'factor' : factor}

    def transform(self, image, factor):
        affine_matrix = np.array([[factor, 0, (1-factor)*self.im_size/2],[0, 1, 0]])
        return super().transform(image, affine_matrix)

    def convert_to_numpy(self, params):
        return np.array([params['factor']])

    def convert_from_numpy(self, numpy_record):
        return {'factor' : numpy_record[0]}


class ScaleY(Affine):

    tags = ['pil', 'spatial', 'scale_y']
    name = 'scale_y'

    def sample_parameters(self):
        factor = float_parameter(sample_level(self.severity,self.max_intensity), 0.5)
        if np.random.uniform() > 0.5:
            factor = -factor
        factor = 2 ** factor
        return {'factor' : factor}

    def transform(self, image, factor):
        affine_matrix = np.array([[1, 0, 0],[0, factor, (1-factor)*self.im_size/2]])
        return super().transform(image, affine_matrix)

    def convert_to_numpy(self, params):
        return np.array([params['factor']])

    def convert_from_numpy(self, numpy_record):
        return {'factor' : numpy_record[0]}

class ScaleFixedAspectRatio(Affine):

    tags = ['pil', 'spatial', 'scale_fixed_aspect_ratio']
    name = 'scale_fixed_aspect_ratio'

    def sample_parameters(self):
        factor = float_parameter(sample_level(self.severity,self.max_intensity), 0.5)
        if np.random.uniform() > 0.5:
            factor = -factor
        factor = 2 ** factor
        return {'factor' : factor}

    def transform(self, image, factor):
        affine_matrix = np.array([[factor, 0, (1-factor)*self.im_size/2],[0, factor, (1-factor)*self.im_size/2]])
        return super().transform(image, affine_matrix)

    def convert_to_numpy(self, params):
        return np.array([params['factor']])

    def convert_from_numpy(self, numpy_record):
        return {'factor' : numpy_record[0]}

class Quadrilateral(Augmentation):

    tags = ['pil', 'spatial', 'quadrilateral']
    name = 'quadrilateral'

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
        return np.array(im)

    def convert_to_numpy(self, params):
        return params['points'].flatten()

    def convert_from_numpy(self, numpy_record):
        return {'points' : numpy_record.reshape(4,2)}

class QuadrilateralNoBars(Augmentation):

    tags = ['pil', 'spatial', 'quadrilateral_no_bars']
    name = 'quadrilateral_no_bars'

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

class KeystoneH(Quadrilateral):

    tags = ['pil', 'spatial', 'keystone_h']
    name = 'keystone_h'

    def sample_parameters(self):
        shift = float_parameter(sample_level(self.severity,self.max_intensity), self.im_size / 3)
        if np.random.uniform() > 0.5:
            shift = - shift
        return {'shift' : shift}

    def transform(self, image, shift):
        points = np.array([
            [0,shift],
            [0, self.im_size-shift],
            [self.im_size, self.im_size+shift],
            [self.im_size, -shift],
            ])
        return super().transform(image, points)

    def convert_to_numpy(self, params):
        return np.array([params['shift']])

    def convert_from_numpy(self, numpy_record):
        return {'shift' : numpy_record[0]}

class KeystoneV(Quadrilateral):

    tags = ['pil', 'spatial', 'keystone_v']
    name = 'keystone_v'

    def sample_parameters(self):
        shift = float_parameter(sample_level(self.severity,self.max_intensity), self.im_size / 3)
        if np.random.uniform() > 0.5:
            shift = - shift
        return {'shift' : shift}

    def transform(self, image, shift):
        points = np.array([
            [shift,0],
            [-shift, self.im_size],
            [self.im_size+shift, self.im_size],
            [self.im_size-shift, 0]
            ])
        return super().transform(image, points)

    def convert_to_numpy(self, params):
        return np.array([params['shift']])

    def convert_from_numpy(self, numpy_record):
        return {'shift' : numpy_record[0]}

class Perspective(Augmentation):

    tags = ['pil', 'spatial', 'perspective']
    name = 'perspective'

    def sample_parameters(self):
        offset_x = float_parameter(sample_level(self.severity,self.max_intensity), 0.1)
        if np.random.uniform() > 0.5:
            offset_x = -offset_x
        offset_y = float_parameter(sample_level(self.severity,self.max_intensity), 0.1)
        if np.random.uniform() > 0.5:
            offset_y = -offset_y
        shift_x = float_parameter(sample_level(self.severity,self.max_intensity), self.im_size / 10)
        #shift_x = 0.0
        if np.random.uniform() > 0.5:
            shift_x = -shift_x
        shift_y = float_parameter(sample_level(self.severity,self.max_intensity), self.im_size / 10)
        #shift_y = 0.0
        if np.random.uniform() > 0.5:
            shift_y = -shift_y
        factor_x = float_parameter(sample_level(self.severity,self.max_intensity), 0.15)
        if np.random.uniform() > 0.5:
            factor_x = -factor_x
        factor_x = 2 ** factor_x
        factor_y = float_parameter(sample_level(self.severity,self.max_intensity), 0.15)
        if np.random.uniform() > 0.5:
            factor_y = -factor_y
        factor_y = 2 ** factor_y
        denom_x = float_parameter(sample_level(self.severity,self.max_intensity), 0.2 / self.im_size)
        if np.random.uniform() > 0.5:
            denom_x = denom_x
        denom_y = float_parameter(sample_level(self.severity,self.max_intensity), 0.2 / self.im_size)
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
        return np.array(im)

    def convert_to_numpy(self, params):
        return params['perspective_params']

    def convert_from_numpy(self, numpy_record):
        return {'perspective_params' : numpy_record}



class PerspectiveNoBars(Augmentation):

    tags = ['pil', 'spatial', 'perspective_no_bars']
    name = 'perspective_no_bars'

    def sample_parameters(self):
        offset_x = float_parameter(sample_level(self.severity,self.max_intensity), 0.1)
        if np.random.uniform() > 0.5:
            offset_x = -offset_x
        offset_y = float_parameter(sample_level(self.severity,self.max_intensity), 0.1)
        if np.random.uniform() > 0.5:
            offset_y = -offset_y
        shift_x = float_parameter(sample_level(self.severity,self.max_intensity), self.im_size / 10)
        #shift_x = 0.0
        if np.random.uniform() > 0.5:
            shift_x = -shift_x
        shift_y = float_parameter(sample_level(self.severity,self.max_intensity), self.im_size / 10)
        #shift_y = 0.0
        if np.random.uniform() > 0.5:
            shift_y = -shift_y
        factor_x = float_parameter(sample_level(self.severity,self.max_intensity), 0.15)
        if np.random.uniform() > 0.5:
            factor_x = -factor_x
        factor_x = 2 ** factor_x
        factor_y = float_parameter(sample_level(self.severity,self.max_intensity), 0.15)
        if np.random.uniform() > 0.5:
            factor_y = -factor_y
        factor_y = 2 ** factor_y
        denom_x = float_parameter(sample_level(self.severity,self.max_intensity), 0.2 / self.im_size)
        if np.random.uniform() > 0.5:
            denom_x = denom_x
        denom_y = float_parameter(sample_level(self.severity,self.max_intensity), 0.2 / self.im_size)
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
