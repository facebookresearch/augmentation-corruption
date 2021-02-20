# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import numpy as np


def is_iterable(obj):
    try:
        iter(obj)
    except:
        return False
    else:
        return True

class Augmentation(abc.ABC):

    tags = ["abstract_base_class"]

    def __init__(self, severity, im_size, record=False, max_intensity=False, **kwargs):
        self.im_size = im_size
        self.severity = severity
        self.record = record
        self.max_intensity = max_intensity

    @abc.abstractmethod
    def transform(self, image, **kwargs):
        ...

    @abc.abstractmethod
    def sample_parameters(self):
        ...

    def __call__(self, image):
        params = self.sample_parameters()
        out = self.transform(image, **params)
        if self.record:
            return out, params
        return out


    def convert_to_numpy(self, params):
        out = []
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                out.extend(v.flatten().tolist())
            elif is_iterable(v):
                out.extend([x for x in v])
            else:
                out.append(v)
        return np.array(out)

    def convert_from_numpy(self, numpy_record):
        param_signature = self.sample_parameters()
        #assert len(param_signature.keys())<=len(numpy_record), "Mismatched numpy_record."
        offset = 0
        for k, v in param_signature.items():
            if isinstance(v, np.ndarray):
                num = len(v.flatten())
                data = numpy_record[offset:offset+num]
                if v.dtype==np.int or v.dtype==np.uint:
                    data = np.round(data, 3)
                data = data.astype(v.dtype)
                param_signature[k] = data.reshape(v.shape)
                offset += num
            elif is_iterable(v):
                data = []
                for x in v:
                    if type(x) == 'int':
                        data.append(int(np.round(numpy_record[offset],3)))
                    else:
                        data.append(type(x)(numpy_record[offset]))
                    offset += 1
                param_signature[k] = data
            else:
                if type(v) == 'int':
                    param_signature[k] = int(np.round(numpy_record[offset],3))
                else:
                    param_signature[k] = type(v)(numpy_record[offset])
                offset += 1
        return param_signature
