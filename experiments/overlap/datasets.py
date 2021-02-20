# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import augmentations as aug
from .augmentations.utils.converters import NumpyToTensor, PilToNumpy
from .augmentations.utils.aug_finder import get_augs_by_tag, parse_aug_string, get_aug_by_name
from .augmentations.utils.severity import sample_level, int_parameter, float_parameter
from .augmentations import pil, compositions, obscure, patch_gaussian, standard_augmentations
import torchvision as tv
import torch
import numpy as np
import os
from PIL import Image, ImageOps

CIFAR_MEAN = [125.3/255, 123.0/255, 113.9/255]
CIFAR_STD = [63.0/255, 62.1/255, 66.7/255]

#This is in RGB order since that is the standard for PIL
IM_MEAN = [0.485, 0.456, 0.406]
IM_STD = [0.229, 0.224, 0.225]

class Cifar10Base(torch.utils.data.Dataset):
    def __init__(self, data_path, split, im_size, train_aug=None, num_transforms=None, augmentation=None, transform_file=None):
        assert split in ['train', 'test'], "Unknown split {}".format(split)
        self.train = True if split=='train' else False
        self.train_aug = self.train if train_aug is None else train_aug
        self.transform_weights = None

        if self.train_aug:
            train_transform = [
                    tv.transforms.RandomHorizontalFlip(),
                    tv.transforms.RandomCrop(im_size, padding=4)
                    ]
        else:
            train_transform = []

        self.pretransform = tv.transforms.Compose(train_transform + [PilToNumpy()])
        if augmentation is None:
            self.aug =  aug.identity.Identity()
        else:
            self.aug = augmentation
        self.posttransform = tv.transforms.Compose([
            NumpyToTensor(),
            tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])

        if transform_file:
            transforms = np.load(transform_file)
            self.transform_list = transforms[:num_transforms]\
                    if num_transforms is not None else transforms
        elif num_transforms:
            self.transform_list = self.build_transform_list(num_transforms)
        else:
            self.transform_list = None

        self.dataset = tv.datasets.CIFAR10(data_path, self.train, download=False)


    def build_transform_list(self, num_transforms):
        transforms = [self.aug.convert_to_numpy(self.aug.sample_parameters()) for i in range(num_transforms)]
        return np.stack(transforms, axis=0)

    def get_random_transform(self):
        if self.transform_list is None:
            return self.aug.sample_parameters()
        elif self.transform_weights is None:
            params = self.transform_list[np.random.randint(low=0, high=len(self.transform_list))]
            return self.aug.convert_from_numpy(params)
        else:
            index = np.random.choice(np.arange(len(self.transform_list)), p=self.transform_weights)
            params = self.transform_list[index]
            return self.aug.convert_from_numpy(params)

    def __getitem__(self, index):
        pre_im, label = self.dataset[index]
        pre_im = self.pretransform(pre_im)
        params = self.get_random_transform()
        return self.posttransform(self.aug.transform(pre_im, **params)), label

    def __len__(self):
        return len(self.dataset)


    def fixed_transform(self, index, transform_index):
        assert self.transform_list is not None, "Must have a fixed transform list to generate fixed transforms."
        im, label = self.dataset[index]
        im = self.pretransform(im)
        params = self.aug.convert_from_numpy(self.transform_list[transform_index])
        im = self.aug.transform(im, **params)
        return self.posttransform(im), label
        
    def serialize(self, indices=None):
        '''
        Returns a new dataset that is all fixed transforms in order,
        applied to each index in order.
        '''
        class SerialDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, indices=None):
                self.dataset = dataset
                self.indices = indices

            def __getitem__(self, index):
                im_idx = index // len(self.dataset.transform_list)
                im_idx = self.indices[im_idx] if self.indices is not None else im_idx
                param_idx = index % len(self.dataset.transform_list)
                return self.dataset.fixed_transform(im_idx, param_idx)

            def __len__(self):
                if self.indices is not None:
                    return len(self.indices) * len(self.dataset.transform_list)
                else:
                    return len(self.dataset) * len(self.dataset.transform_list)
        return SerialDataset(self, indices)

class Cifar10Augmix(Cifar10Base):
    def __init__(self, data_path, split, im_size, train_aug=None, num_transforms=None,
            aug_string=None, width=3, depth=3, random_depth=True, prob_coeff=1.0,
            severity=3, transform_file=None):

        self.aug_string = aug_string
        self.width = width
        self.depth = depth
        self.prob_coeff = prob_coeff
        self.random_depth = random_depth
        self.severity = severity
        if aug_string is not None:
            augs = parse_aug_string(aug_string, im_size)
        else:
            augs = get_augs_by_tag(['augmix'])
            augs = [a(severity=severity, im_size=im_size) for a in augs]
        augmentation = compositions.Augmix(
                augmentation_list=augs,
                width=width,
                max_depth=depth,
                random_depth=random_depth,
                prob_coeff=prob_coeff
                )
        super(Cifar10Augmix, self).__init__(data_path, split, im_size, train_aug, num_transforms,
                augmentation, transform_file=transform_file)

        
class Cifar10RandomSample(Cifar10Base):
    def __init__(self, data_path, split, im_size, train_aug=None, num_transforms=None,
            aug_string=None, severity=3, weights=None):
        self.aug_string = aug_string
        if aug_string is not None:
            augs = parse_aug_string(aug_string, im_size)
        else:
            augs = get_augs_by_tag(['augmix'])
            augs = [a(severity=severity, im_size=im_size) for a in augs]
        augmentation = compositions.RandomSample(
                augmentation_list=augs,
                weights=weights
                )
        super(Cifar10RandomSample, self).__init__(data_path, split, im_size, train_aug, num_transforms, augmentation)




class Cifar10Corruption(Cifar10Base):
    '''
    Corruptions are different in three ways: they sample at fixed max intensity
    of randomly between a low value and some maximum, they generate
    fixed transforms in order and balanced (and can give the corruption/severity
    of a given transform index), and have the metadata for the frost corruption.
    '''
    def __init__(self, data_path, split, im_size, train_aug=None, num_transforms=None,
            aug_string=None, frost_path=None, include_extra=True, random_transforms=False):
        self.include_extra = include_extra
        self.random_transforms = random_transforms
        if aug_string is not None:
            self.aug_list = parse_aug_string(aug_string, im_size, max_intensity=True, frost_path=frost_path)
        else:
            augs = get_augs_by_tag(['imagenet_c'], [] if include_extra else ['extra'])
            severities = [1,2,3,4,5]
            self.aug_list = [a(severity=s, im_size=im_size, max_intensity=True, frost_path=frost_path)\
                    for a in augs for s in severities]
        augmentation = compositions.RandomSample(
                augmentation_list=self.aug_list
                )
        
        super(Cifar10Corruption, self).__init__(data_path, split, im_size, train_aug, num_transforms, augmentation)

    def build_transform_list(self, num_transforms):
        if self.random_transforms:
            return super(Cifar10Corruption, self).build_transform_list(num_transforms)

        assert num_transforms % len(self.aug_list) == 0,\
            "The total number of augs needs to divide into the total number of transforms."
        transform_list = None
        for i in range(num_transforms):
            transform_idx = i // (num_transforms // len(self.aug_list))
            transform_params = self.aug_list[transform_idx].sample_parameters()
            curr_record = self.aug.convert_to_numpy({
                'idx' : transform_idx,
                'transform_params' : transform_params
                }).reshape(1,-1)
            transform_list = np.concatenate((transform_list, curr_record), axis=0)\
                    if transform_list is not None else curr_record
        return transform_list

    def get_corruption(self, transform_index):
        aug_type_index = transform_index // (len(self.transform_list) // len(self.aug_list))
        return self.aug_list[aug_type_index].name, self.aug_list[aug_type_index].severity

class Cifar10AutoAugment(Cifar10Base):
    def __init__(self, data_path, split, im_size, train_aug=None, num_transforms=None, subpolicy_list=None, add_cutout=False, transform_file=None):

        def stom(low, high, sev):
            return sev / 10 * (high - low) + low
        
        size = im_size
        init = lambda transform : transform(0, size)
        tn = 150/331 * im_size
        if subpolicy_list is None:
            subpolicy_list = [
                [(init(pil.Invert), 0.1, None, None), (init(pil.Contrast), 0.2, stom(0,0.9,6), 1)],
                [(init(pil.Rotate), 0.7, stom(0,30,2), 0), (init(pil.TranslateX), 0.3, stom(0,tn,9), 0)],
                [(init(pil.Sharpness), 0.8, stom(0,0.9,1), 1), (init(pil.Sharpness), 0.9, stom(0,0.9,3), 1)],
                [(init(pil.ShearY), 0.5, stom(0,0.3,8), 0), (init(pil.TranslateY), 0.7, stom(0,tn,9), 0)],
                [(init(pil.AutoContrast), 0.5, None, None), (init(pil.Equalize), 0.9, None, None)],
                [(init(pil.ShearY), 0.2, stom(0,0.3,7), 0), (init(pil.Posterize), 0.3, int(stom(4,8,7)), None)],
                [(init(pil.ColorBalance), 0.4, stom(0,0.9,3),1), (init(pil.Brightness), 0.6, stom(0,0.9,7),1)],
                [(init(pil.Sharpness), 0.3, stom(0,0.9,9),1), (init(pil.Brightness), 0.7, stom(0,0.9,9),1)],
                [(init(pil.Equalize), 0.6, None, None), (init(pil.Equalize), 0.5, None, None)],
                [(init(pil.Contrast), 0.6, stom(0,0.9,7),1), (init(pil.Sharpness), 0.6, stom(0,0.9,5),1)],
                [(init(pil.ColorBalance), 0.7, stom(0,0.9,7),1), (init(pil.TranslateX), 0.5, stom(0,tn,8),0)],
                [(init(pil.Equalize), 0.3, None, None), (init(pil.AutoContrast), 0.4, None, None)],
                [(init(pil.TranslateY), 0.4, stom(0,tn,3),0), (init(pil.Sharpness), 0.2, stom(0,0.9,6),1)],
                [(init(pil.Brightness), 0.9, stom(0,0.9,6),1), (init(pil.ColorBalance), 0.2, stom(0,0.9,8),1)],
                [(init(pil.Solarize), 0.5, stom(256,0,2),None), (init(pil.Invert), 0.0, None,None)],
                [(init(pil.Equalize), 0.2, None, None), (init(pil.AutoContrast), 0.6, None, None)],
                [(init(pil.Equalize), 0.2, None, None), (init(pil.Equalize), 0.6, None, None)],
                [(init(pil.ColorBalance), 0.9, stom(0,0.9,9),1), (init(pil.Equalize), 0.6, None, None)],
                [(init(pil.AutoContrast), 0.8, None, None), (init(pil.Solarize), 0.2, stom(256,0,8), None)],
                [(init(pil.Brightness), 0.1, stom(0,0.9,3),1), (init(pil.ColorBalance), 0.7, stom(0,0.9,0),1)],
                [(init(pil.Solarize), 0.4, stom(256,0,5), None), (init(pil.AutoContrast), 0.9, None, None)],
                [(init(pil.TranslateY), 0.9, stom(0,tn,9), None), (init(pil.TranslateY), 0.7, stom(0,tn,9),0)],
                [(init(pil.AutoContrast), 0.9, None, None), (init(pil.Solarize), 0.8, stom(256,0,3), None)],
                [(init(pil.Equalize), 0.8, None, None), (init(pil.Invert), 0.1, None, None)],
                [(init(pil.TranslateY), 0.7, stom(0,tn,9), 0), (init(pil.AutoContrast), 0.9, None, None)]
            ]


        aug = compositions.AutoAugment(subpolicy_list)
        if add_cutout:
            cutout = obscure.CutOut(severity=10, im_size=im_size, max_intensity=True)
            aug = compositions.ComposeSerially([aug, cutout])

        super(Cifar10AutoAugment, self).__init__(data_path, split, im_size, train_aug, num_transforms,
                aug, transform_file=transform_file)



class Cifar10PatchGaussian(Cifar10Base):
    def __init__(self, data_path, split, im_size, train_aug=None, num_transforms=None, transform_file=None, patch_width=25, patch_sigma=1.0, max_width=True):
        if patch_width is not None:
            aug = patch_gaussian.PatchGaussian(
                    severity=None,
                    im_size=im_size,
                    max_intensity=max_width,
                    sigma=patch_sigma,
                    width=patch_width
                    )
        else:
            aug = patch_gaussian.Gaussian(
                    severity = patch_sigma * 10,
                    im_size=im_size,
                    max_intensity=max_width
                    )
        if train_aug or (split=='train' and train_aug is None):
            train = standard_augmentations.Cifar10CropAndFlip(severity=None, im_size=im_size)
            aug = compositions.ComposeSerially([aug, train])

        super(Cifar10PatchGaussian, self).__init__(data_path, split, im_size, False, num_transforms,
                aug, transform_file=transform_file)



class ImageNetBase(torch.utils.data.Dataset):
    def __init__(self, data_path, split, im_size, train_aug=None, num_transforms=None, augmentation=None, rgb_to_bgr=True):
        assert split in ['train', 'val'], "Unknown split {}".format(split)
        self.train = True if split=='train' else False
        self.train_aug = self.train if train_aug is None else train_aug
        data_path = os.path.join(data_path, split)

        if self.train_aug:
            train_transform = [
                    tv.transforms.RandomResizedCrop(im_size, scale=(0.08,1.0)),
                    tv.transforms.RandomHorizontalFlip(),
                    ]
        else:
            train_transform = [
                    tv.transforms.Resize(256),
                    tv.transforms.CenterCrop(im_size)
                    ]

        def RGB_to_BGR(image):
            return image[[2,1,0],:,:]

        self.pretransform = tv.transforms.Compose(train_transform + [PilToNumpy()])
        if augmentation is None:
            self.aug =  aug.identity.Identity()
        else:
            self.aug = augmentation
        self.posttransform = tv.transforms.Compose([
            NumpyToTensor(),
            tv.transforms.Normalize(IM_MEAN, IM_STD)] +
            ([RGB_to_BGR] if rgb_to_bgr else []) #PyCls imagenet models are trained in BGR input order
            )
        

        self.transform_list = self.build_transform_list(num_transforms)\
                if num_transforms is not None else None

        self.dataset = tv.datasets.ImageFolder(data_path, None)


    def build_transform_list(self, num_transforms):
        transforms = [self.aug.convert_to_numpy(self.aug.sample_parameters()) for i in range(num_transforms)]
        return np.stack(transforms, axis=0)

    def get_random_transform(self):
        if self.transform_list is None:
            return self.aug.sample_parameters()
        else:
            params = self.transform_list[np.random.randint(low=0, high=len(self.transform_list))]
            return self.aug.convert_from_numpy(params)

    def __getitem__(self, index):
        pre_im, label = self.dataset[index]
        pre_im = self.pretransform(pre_im)
        params = self.get_random_transform()
        return self.posttransform(self.aug.transform(pre_im, **params)), label

    def __len__(self):
        return len(self.dataset)


    def fixed_transform(self, index, transform_index):
        assert self.transform_list is not None, "Must have a fixed transform list to generate fixed transforms."
        im, label = self.dataset[index]
        im = self.pretransform(im)
        params = self.aug.convert_from_numpy(self.transform_list[transform_index])
        im = self.aug.transform(im, **params)
        return self.posttransform(im), label
        
    def serialize(self, indices=None):
        '''
        Returns a new dataset that is all fixed transforms in order,
        applied to each index in order.
        '''
        class SerialDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, indices=None):
                self.dataset = dataset
                self.indices = indices

            def __getitem__(self, index):
                im_idx = index // len(self.dataset.transform_list)
                im_idx = self.indices[im_idx] if self.indices is not None else im_idx
                param_idx = index % len(self.dataset.transform_list)
                return self.dataset.fixed_transform(im_idx, param_idx)

            def __len__(self):
                if self.indices is not None:
                    return len(self.indices) * len(self.dataset.transform_list)
                else:
                    return len(self.dataset) * len(self.dataset.transform_list)
        return SerialDataset(self, indices)

class ImageNetCorruption(ImageNetBase):
    '''
    Corruptions are different in three ways: they sample at fixed max intensity
    of randomly between a low value and some maximum, they generate
    fixed transforms in order and balanced (and can give the corruption/severity
    of a given transform index), and have the metadata for the frost corruption.
    '''
    def __init__(self, data_path, split, im_size, train_aug=None, num_transforms=None,
            aug_string=None, frost_path=None, include_extra=True, rgb_to_bgr=True):
        self.include_extra = include_extra
        if aug_string is not None:
            self.aug_list = parse_aug_string(aug_string, im_size, max_intensity=True, frost_path=frost_path)
        else:
            augs = get_augs_by_tag(['imagenet_c'], [] if include_extra else ['extra'])
            severities = [1,2,3,4,5]
            self.aug_list = [a(severity=s, im_size=im_size, max_intensity=True, frost_path=frost_path)\
                    for a in augs for s in severities]
        augmentation = compositions.RandomSample(
                augmentation_list=self.aug_list
                )
        
        super(ImageNetCorruption, self).__init__(data_path, split, im_size, train_aug, num_transforms,
                augmentation, rgb_to_bgr)

    def build_transform_list(self, num_transforms):
        assert num_transforms % len(self.aug_list) == 0,\
            "The total number of augs needs to divide into the total number of transforms."
        transform_list = None
        for i in range(num_transforms):
            transform_idx = i // (num_transforms // len(self.aug_list))
            transform_params = self.aug_list[transform_idx].sample_parameters()
            curr_record = self.aug.convert_to_numpy({
                'idx' : transform_idx,
                'transform_params' : transform_params
                }).reshape(1,-1)
            transform_list = np.concatenate((transform_list, curr_record), axis=0)\
                    if transform_list is not None else curr_record
        return transform_list

    def get_corruption(self, transform_index):
        aug_type_index = transform_index // (len(self.transform_list) // len(self.aug_list))
        return self.aug_list[aug_type_index].name, self.aug_list[aug_type_index].severity



class ImageNetPatchGaussian(ImageNetBase):
    def __init__(self, data_path, split, im_size, train_aug=None, num_transforms=None, patch_width=250, patch_sigma=1.0, max_width=False, rgb_to_bgr=True):
        if patch_width is not None:
            aug = patch_gaussian.PatchGaussian(
                    severity=None,
                    im_size=im_size,
                    max_intensity=max_width,
                    sigma=patch_sigma,
                    width=patch_width
                    )
        else:
            aug = patch_gaussian.Gaussian(
                    severity = patch_sigma * 10,
                    im_size=im_size,
                    max_intensity=max_width
                    )

        super(ImageNetPatchGaussian, self).__init__(data_path, split, im_size, train_aug, num_transforms,
                aug, rgb_to_bgr=rgb_to_bgr)



class ImageNetAutoAugment(ImageNetBase):
    def __init__(self, data_path, split, im_size, train_aug=None, num_transforms=None, subpolicy_list=None, rgb_to_bgr=True):

        def stom(low, high, sev):
            return sev / 10 * (high - low) + low
        
        size = im_size
        init = lambda transform : transform(0, size)
        tn = 150/331 * im_size
        if subpolicy_list is None:
            subpolicy_list = [

                [(init(pil.Posterize), 0.4, int(stom(4,8,8)), None), (init(pil.Rotate), 0.6, stom(0,30,9),0)],
                [(init(pil.Solarize), 0.6, stom(256,0,5), None), (init(pil.AutoContrast), 0.6, None,None)],
                [(init(pil.Equalize), 0.8, None, None), (init(pil.Equalize), 0.6, None, None)],
                [(init(pil.Posterize), 0.6, int(stom(4,8,7)), None), (init(pil.Posterize), 0.6, int(stom(4,8,6)),None)],
                [(init(pil.Equalize), 0.4, None, None), (init(pil.Solarize), 0.2, stom(256,0,4),None)],

                [(init(pil.Equalize), 0.4, None, None), (init(pil.Rotate), 0.8, stom(0,30,8),0)],
                [(init(pil.Solarize), 0.6, stom(256,0,3), None), (init(pil.Equalize), 0.6, None, None)],
                [(init(pil.Posterize), 0.8, int(stom(4,8,5)), None), (init(pil.Equalize), 1.0, None, None)],
                [(init(pil.Rotate), 0.2, stom(0,30,3), 0), (init(pil.Solarize), 0.6, stom(256,0,8),None)],
                [(init(pil.Equalize), 0.6, None, None), (init(pil.Posterize), 0.4, int(stom(4,8,6)),None)],
                [(init(pil.Rotate), 0.8, stom(0,30,8), 0), (init(pil.ColorBalance), 0.4, stom(0,0.9,0),1)],
                [(init(pil.Rotate), 0.4, stom(0,30,9), 0), (init(pil.Equalize), 0.6, None, None)],
                [(init(pil.Equalize), 0.0, None, None), (init(pil.Equalize), 0.8, None, None)],
                [(init(pil.Invert), 0.6, None, None), (init(pil.Equalize), 1.0, None, None)],
                [(init(pil.ColorBalance), 0.6, stom(0,0.9,4), 1), (init(pil.Contrast), 1.0, stom(0,0.9,8),1)],
                [(init(pil.Rotate), 0.8, stom(0,30,8), 0), (init(pil.ColorBalance), 1.0, stom(0,0.9,2),1)],
                [(init(pil.ColorBalance), 0.8, stom(0,0.9,8), 1), (init(pil.Solarize), 0.8, stom(256,0,7),None)],
                [(init(pil.Sharpness), 0.4, stom(0,0.9,7), 1), (init(pil.Invert), 0.6, None, None)],
                [(init(pil.ShearX), 0.6, stom(0,0.9,5), 1), (init(pil.Equalize), 1.0, None, None)],
                [(init(pil.ColorBalance), 0.4, stom(0,0.9,0), 1), (init(pil.Equalize), 0.6, None, None)],
                [(init(pil.Equalize), 0.4, None, None), (init(pil.Solarize), 0.2, stom(256,0,4),None)],
                [(init(pil.Solarize), 0.6, stom(256,0,5), None), (init(pil.AutoContrast), 0.6, None, None)],
                [(init(pil.Invert), 0.6, None, None), (init(pil.Equalize), 1.0, None, None)],
                [(init(pil.ColorBalance), 0.6, stom(0,0.9,4), 1), (init(pil.Contrast), 1.0, stom(0,0.9,8),1)],
                [(init(pil.Equalize), 0.8, None, None), (init(pil.Equalize), 0.6, None, None)],
                
            ]


        aug = compositions.AutoAugment(subpolicy_list)

        super(ImageNetAutoAugment, self).__init__(data_path, split, im_size, train_aug, num_transforms,
                aug, rgb_to_bgr=rgb_to_bgr)


class ImageNetAugmix(ImageNetBase):
    def __init__(self, data_path, split, im_size, train_aug=None, num_transforms=None,
            aug_string=None, width=3, depth=3, random_depth=True, prob_coeff=1.0,
            severity=3, rgb_to_bgr=True):

        self.aug_string = aug_string
        self.width = width
        self.depth = depth
        self.prob_coeff = prob_coeff
        self.random_depth = random_depth
        self.severity = severity
        if aug_string is not None:
            augs = parse_aug_string(aug_string, im_size)
        else:
            augs = get_augs_by_tag(['augmix'])
            augs = [a(severity=severity, im_size=im_size) for a in augs]
        augmentation = compositions.Augmix(
                augmentation_list=augs,
                width=width,
                max_depth=depth,
                random_depth=random_depth,
                prob_coeff=prob_coeff
                )
        super(ImageNetAugmix, self).__init__(data_path, split, im_size, train_aug, num_transforms,
                augmentation, rgb_to_bgr=rgb_to_bgr)


class Cifar10AugmixJSD(torch.utils.data.Dataset):

    def __init__(self, data_path, split, im_size, train_aug=True,
            augmix_width=3, augmix_depth=3, augmix_random_depth=True,
            augmix_prob_coeff=1.0, augmix_severity=3,
            jsd_num=3):
        self.jsd_num = jsd_num
        self.split = split
        self.train = True if split=='train' else False

        train_transform = [tv.transforms.RandomHorizontalFlip(),
           tv.transforms.RandomCrop(im_size, padding=4)]\
                   if (self.train and train_aug) else []
        self.pretransform = tv.transforms.Compose(train_transform + [PilToNumpy()])
        self.posttransform = tv.transforms.Compose([NumpyToTensor(), tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

        aug_list = [
                pil.ShearX(augmix_severity, im_size=im_size),
                pil.ShearY(augmix_severity, im_size=im_size),
                pil.TranslateX(augmix_severity, im_size=im_size),
                pil.TranslateY(augmix_severity, im_size=im_size),
                pil.Rotate(augmix_severity, im_size=im_size),
                pil.Equalize(augmix_severity, im_size=im_size),
                pil.AutoContrast(augmix_severity, im_size=im_size),
                pil.Solarize(augmix_severity, im_size=im_size),
                pil.Posterize(augmix_severity, im_size=im_size)
                ]

        self.aug = compositions.Augmix(
                augmentation_list=aug_list,
                width=augmix_width,
                max_depth=augmix_depth,
                random_depth=augmix_random_depth,
                prob_coeff=augmix_prob_coeff
                )

        self.dataset = tv.datasets.CIFAR10(data_path, self.train, transform=None, download=False)

    def __getitem__(self, index):
        im, label = self.dataset[index]
        im = self.pretransform(im)
        im_one = self.posttransform(im)
        ims = [self.posttransform(self.aug(im)) for i in range(self.jsd_num-1)]
        c, h, w = im_one.size()
        out = torch.stack([im_one] + ims, dim=0).view(c * self.jsd_num, h, w)
        return out, label

    def __len__(self):
        return len(self.dataset)

class ImageNetAugmixJSD(torch.utils.data.Dataset):

    def __init__(self, data_path, split, im_size, RGB_to_BGR=True, mixture_width=3, mixture_depth=-1, aug_severity=1, aug_prob_coeff=1, jsd_num=3):
        self.split = split
        self.train = True if split=='train' else False
        self.im_size = im_size
        self.RGB_to_BGR = RGB_to_BGR

        self.train_transform = tv.transforms.Compose(
              [tv.transforms.RandomResizedCrop(im_size, scale=(0.08,1.0)),
               tv.transforms.RandomHorizontalFlip()])
        self.test_transform = tv.transforms.Compose(
              [tv.transforms.Resize(256),
               tv.transforms.CenterCrop(im_size)])
        self.preprocess = tv.transforms.Compose(
              [tv.transforms.ToTensor(),
               tv.transforms.Normalize(IM_MEAN, IM_STD)])
        data_path = os.path.join(data_path, split)
        self.transform = self.train_transform if self.train else self.test_transform
        self.dataset = tv.datasets.ImageFolder(data_path, None)

        self.width = mixture_width
        self.depth = mixture_depth
        self.severity = aug_severity
        self.prob_coeff = aug_prob_coeff
        self.im_size = im_size
        self.num = jsd_num
        self.augmentations = [
                self.rotate,
                self.shear_x,
                self.shear_y,
                self.translate_x,
                self.translate_y,
                self.autocontrast,
                self.posterize,
                self.equalize,
                self.solarize,
                ]

    def _prepare_im(self, im):
        im = self.preprocess(im)
        if self.RGB_to_BGR:
            im = im[[2,1,0],:,:]
        return im

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im, label = self.dataset[index]
        im = self.transform(im)
        ims = [self._prepare_im(im)] + [self.augment(im) for i in range(1,self.num)]
        im = np.concatenate(ims, axis=0)
        return im, label

    def augment(self, im):
        ws = np.float32(
        np.random.dirichlet([self.prob_coeff] * self.width))
        m = np.float32(np.random.beta(self.prob_coeff, self.prob_coeff))

        mix = torch.zeros_like(self._prepare_im(im))
        for i in range(self.width):
            image_aug = im.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = op(image_aug, self.severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * self._prepare_im(image_aug)

        mixed = (1 - m) * self._prepare_im(im) + m * mix
        return mixed

    def autocontrast(self, pil_img, _):
        return ImageOps.autocontrast(pil_img)

    def equalize(self, pil_img, _):
        return ImageOps.equalize(pil_img)

    def posterize(self, pil_img, level):
        level = int_parameter(sample_level(level), 4)
        return ImageOps.posterize(pil_img, 4 - level)

    def rotate(self, pil_img, level):
        degrees = int_parameter(sample_level(level), 30)
        if np.random.uniform() > 0.5:
            degrees = -degrees
        return pil_img.rotate(degrees, resample=Image.BILINEAR)

    def solarize(self, pil_img, level):
        level = int_parameter(sample_level(level), 256)
        return ImageOps.solarize(pil_img, 256 - level)

    def shear_x(self, pil_img, level):
        level = float_parameter(sample_level(level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.im_size, self.im_size),
                               Image.AFFINE, (1, level, 0, 0, 1, 0),
                               resample=Image.BILINEAR)

    def shear_y(self, pil_img, level):
        level = float_parameter(sample_level(level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.im_size, self.im_size),
                               Image.AFFINE, (1, 0, 0, level, 1, 0),
                               resample=Image.BILINEAR)

    def translate_x(self, pil_img, level):
        level = int_parameter(sample_level(level), self.im_size / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.im_size, self.im_size),
                               Image.AFFINE, (1, 0, level, 0, 1, 0),
                               resample=Image.BILINEAR)

    def translate_y(self, pil_img, level):
        level = int_parameter(sample_level(level), self.im_size / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.im_size, self.im_size),
                               Image.AFFINE, (1, 0, 0, 0, 1, level),
                               resample=Image.BILINEAR)

class ImageNetSIN(torch.utils.data.Dataset):
    def __init__(self, in_data_path, sin_data_path, split, im_size, train_aug=None, rgb_to_bgr=True):
        assert split in ['train', 'val'], "Unknown split {}".format(split)
        self.train = True if split=='train' else False
        self.train_aug = self.train if train_aug is None else train_aug
        in_data_path = os.path.join(in_data_path, split)
        sin_data_path = os.path.join(sin_data_path, split)

        if self.train_aug:
            train_transform = [
                    tv.transforms.RandomResizedCrop(im_size, scale=(0.08,1.0)),
                    tv.transforms.RandomHorizontalFlip(),
                    ]
        else:
            train_transform = [
                    tv.transforms.Resize(256),
                    tv.transforms.CenterCrop(im_size)
                    ]

        def RGB_to_BGR(image):
            return image[[2,1,0],:,:]

        self.pretransform = tv.transforms.Compose(train_transform + [PilToNumpy()])
        self.posttransform = tv.transforms.Compose([
            NumpyToTensor(),
            tv.transforms.Normalize(IM_MEAN, IM_STD)] +
            ([RGB_to_BGR] if rgb_to_bgr else []) #PyCls imagenet models are trained in BGR input order
            )
        self.transform = tv.transforms.Compose([
            self.pretransform,
            self.posttransform
            ])

        self.dataset = torch.utils.data.ConcatDataset([
            tv.datasets.ImageFolder(in_data_path, self.transform),
            tv.datasets.ImageFolder(sin_data_path, self.transform)
            ])

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
