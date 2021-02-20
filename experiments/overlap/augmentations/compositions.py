# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import Augmentation
from collections import namedtuple
import numpy as np

class Augmix(Augmentation):

    tags = ['compositor', 'augmix_compose']

    def __init__(self, severity=None, im_size=None, augmentation_list=[], width=3, max_depth=3, prob_coeff=1.0, random_depth=True, record=False, float_output=True):
        super(Augmix, self).__init__(severity, im_size, record)
        self.width = width
        self.depth = max_depth
        self.random_depth = random_depth
        self.prob_coeff = prob_coeff
        self.augs = augmentation_list
        self.float_output = float_output
        self.record_length = max([len(a.convert_to_numpy(a.sample_parameters())) for a in self.augs])\
                if self.augs else 0

    def transform(self, image, m, ws, aug_record):
        if not self.augs:
            return image

        mix = np.zeros_like(image).astype(np.float32)
        for i in range(self.width):
            image_aug = image.copy()
            for j in range(self.depth):
                pos = self.depth * i + j
                if aug_record[pos].idx < 0:
                    continue
                op = self.augs[aug_record[pos].idx].transform
                image_aug = op(image_aug, **(aug_record[pos].params))
            mix += ws[i] * image_aug.astype(np.float32)
        mixed = (1 - m) * image.astype(np.float32) + m * mix

        if self.float_output:
            return mixed
        return mixed.astype(np.uint8)


    def sample_parameters(self):
        ws = np.float32(np.random.dirichlet([self.prob_coeff] * self.width))
        m = np.float32(np.random.beta(self.prob_coeff, self.prob_coeff))
        if not self.augs:
            return { 'm' : m, 'ws' : ws, 'aug_record': []}

        aug_idxs = np.random.randint(low=0, high=len(self.augs), size=self.width*self.depth)
        if self.random_depth:
            for i in range(self.width):
                inverse_depth = np.random.randint(1,self.depth+1)
                aug_idxs[self.depth*i+inverse_depth:self.depth*(i+1)] = -1

        aug_params = [self.augs[i].sample_parameters() if i != -1 else {} for i in aug_idxs]

        AugRecord = namedtuple('AugRecord', ('idx', 'params'))
        return { 'm' : m, 'ws' : ws, 'aug_record' : [AugRecord(idx, params) for idx, params in zip(aug_idxs, aug_params)]}

    def convert_from_numpy(self, record):
        out = {}
        out['m'] = record[0]
        out['ws'] = record[1:self.width+1]
        if not self.augs:
            out['aug_record'] = {}
            return out
        idxs = record[self.width+1:self.width+1+self.width*self.depth]
        params = []
        for i,idx in enumerate(idxs):
            offset = self.width+1+self.width*self.depth + i * self.record_length
            if idx < 0:
                params.append({})
                continue
            sub_params = self.augs[int(idx)].convert_from_numpy(record[offset:offset+self.record_length])
            params.append(sub_params)
        AugRecord = namedtuple('AugRecord', ('idx', 'params'))
        out['aug_record'] = [AugRecord(int(idx), params) for idx, params in zip(idxs, params)]
        return out 

    def convert_to_numpy(self, record):
        out = np.zeros(1+self.width+(self.width*self.depth*(self.record_length+1)))
        if not self.augs:
            return out
        out[0] = record['m']
        out[1:self.width+1] = record['ws']
        sub_record = record['aug_record']
        out[self.width+1:self.width+1+self.width*self.depth] = [i.idx for i in sub_record]
        param_list = []
        for a in record['aug_record']:
            if a.idx >= 0:
                curr_params = self.augs[a.idx].convert_to_numpy(a.params)
                if len(curr_params) < self.record_length:
                    curr_params = np.concatenate((curr_params, np.zeros(self.record_length-len(curr_params))))
            else:
                curr_params = np.zeros(self.record_length)
            param_list.append(curr_params)
        params = np.concatenate(param_list)
        out[self.width+1+self.width*self.depth:] = params
        return out 



class AutoAugmentOld(Augmentation):

    tags = ['compositor', 'autoaugment_compose']

    def __init__(self, subpolicies, severity=None, im_size=None, record=False):
        super(AutoAugmentOld, self).__init__(severity, im_size, record)
        self.subpolicies = subpolicies
        self.record_length = 1+2*max([len(policy) for policy in self.subpolicies])

    def sample_parameters(self):
        policy_idx = np.random.randint(low=0, high=len(self.subpolicies))
        selected = np.random.uniform(low=0.0, high=1.0, size=len(self.subpolicies[policy_idx]))
        thresholds = np.array([transform_tuple[1] for transform_tuple in self.subpolicies[policy_idx]])
        selected = (selected < thresholds).tolist()
        flipped = [(np.random.choice([1,-1]) if (selected[i]==True and p[2] is not None and p[2]<0) else 1) for i,p in enumerate(self.subpolicies[policy_idx])]
        return { 'policy_idx' : policy_idx, 'selections' : selected, 'flipped' : flipped }

    def transform(self, image, policy_idx, selections, flipped):
        policy = self.subpolicies[policy_idx]
        for i, transform_tuple in enumerate(policy):
            if selections[i]:
                transform = transform_tuple[0]
                magnitude = transform_tuple[2]
                if magnitude is not None:
                    image = transform.transform(image, magnitude * flipped[i])
                else:
                    image = transform.transform(image)
        return image

    def convert_to_numpy(self, params):
        out = np.zeros(self.record_length)
        out[0] = params['policy_idx']
        curr_len = len(self.subpolicies[params['policy_idx']])
        out[1:curr_len+1] = params['selections']
        out[1+curr_len:1+2*curr_len] = params['flipped']
        return out

    def convert_from_numpy(self, numpy_record):
        params = {}
        params['policy_idx'] = int(numpy_record[0])
        curr_len = len(self.subpolicies[params['policy_idx']])
        params['selections'] = [True if int(x)==1 else False for x in numpy_record[1:1+curr_len]]
        params['flipped'] = [int(x) for x in numpy_record[1+curr_len:1+2*curr_len]]
        return params
        
class AutoAugment(Augmentation):

    tags = ['compositor', 'autoaugment_compose']

    def __init__(self, subpolicies, severity=None, im_size=None, record=False):
        super(AutoAugment, self).__init__(severity, im_size, record)
        self.subpolicies = subpolicies
        self.record_length = 1+2*max([len(policy) for policy in self.subpolicies])

    def sample_parameters(self):
        policy_idx = np.random.randint(low=0, high=len(self.subpolicies))
        selected = np.random.uniform(low=0.0, high=1.0, size=len(self.subpolicies[policy_idx]))
        thresholds = np.array([transform_tuple[1] for transform_tuple in self.subpolicies[policy_idx]])
        selected = (selected < thresholds).tolist()
        flipped = [(np.random.choice([1,-1]) if (selected[i]==True and p[3] is not None) else 1) for i,p in enumerate(self.subpolicies[policy_idx])]
        return { 'policy_idx' : policy_idx, 'selections' : selected, 'flipped' : flipped }

    def transform(self, image, policy_idx, selections, flipped):
        policy = self.subpolicies[policy_idx]
        for i, transform_tuple in enumerate(policy):
            if selections[i]:
                transform = transform_tuple[0]
                magnitude = transform_tuple[2]
                if magnitude is not None:
                    magnitude = (transform_tuple[3] if transform_tuple[3] is not None else 0) + magnitude * flipped[i]
                    image = transform.transform(image, magnitude)
                else:
                    image = transform.transform(image)
        return image

    def convert_to_numpy(self, params):
        out = np.zeros(self.record_length)
        out[0] = params['policy_idx']
        curr_len = len(self.subpolicies[params['policy_idx']])
        out[1:curr_len+1] = params['selections']
        out[1+curr_len:1+2*curr_len] = params['flipped']
        return out

    def convert_from_numpy(self, numpy_record):
        params = {}
        params['policy_idx'] = int(numpy_record[0])
        curr_len = len(self.subpolicies[params['policy_idx']])
        params['selections'] = [True if int(x)==1 else False for x in numpy_record[1:1+curr_len]]
        params['flipped'] = [int(x) for x in numpy_record[1+curr_len:1+2*curr_len]]
        return params


class RandomSample(Augmentation):
    def __init__(self, augmentation_list, weights=None, severity=None, im_size=None, record=False):
        super(RandomSample, self).__init__(severity=severity, im_size=im_size, record=record)
        self.transforms = augmentation_list
        self.weights = weights
        assert weights is None or (len(weights)==len(augmentation_list)),\
                "Must have equal number of weights as transforms."
        assert weights is None or (np.sum(weights)==1.0),\
                "Weights must sum to one."
        self.record_length = max([len(a.convert_to_numpy(a.sample_parameters())) for a in self.transforms])\
                if self.transforms else 0

    def sample_parameters(self):
        idx = np.random.choice(np.arange(len(self.transforms)), p=self.weights)
        transform_params = self.transforms[idx].sample_parameters()
        return {'idx': idx, 'transform_params': transform_params}

    def transform(self, image, idx, transform_params):
        return self.transforms[idx].transform(image, **transform_params)

    def convert_from_numpy(self, record):
        idx = int(record[0])
        transform_params = self.transforms[idx].convert_from_numpy(record[1:])
        return {'idx' : idx, 'transform_params': transform_params}

    def convert_to_numpy(self, record):
        numpy_record = np.zeros(1+self.record_length)
        numpy_record[0] = record['idx']
        numpy_params = self.transforms[record['idx']].convert_to_numpy(record['transform_params'])
        numpy_record[1:1+len(numpy_params)] = numpy_params
        return numpy_record

class ComposeSerially(Augmentation):
    def __init__(self, augmentation_list, severity=None, im_size=None, record=False):
        self.augmentation_list = augmentation_list
        self.record_lengths = [len(a.convert_to_numpy(a.sample_parameters())) for a in augmentation_list]

    def sample_parameters(self):
        params = {'param_list' : [a.sample_parameters() for a in self.augmentation_list]}
        return params

    def transform(self, image, param_list):
        for a, p in zip(self.augmentation_list, param_list):
            image = a.transform(image, **p)
        return image

    def convert_to_numpy(self, params):
        record = None
        for a, p in zip(self.augmentation_list, params['param_list']):
            record = np.concatenate((record, a.convert_to_numpy(p)), axis=0)\
                   if record is not None else a.convert_to_numpy(p)
        return record

    def convert_from_numpy(self, numpy_record):
        offset = 0
        params = {'params_list' : []}
        for a, d in zip(self.augmentation_list, self.record_lengths):
            params['params_list'].append(numpy_record[offset:offset+d])
            offset += d
        return params
   


