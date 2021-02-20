# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import corrupt as corr

transform_list = [
    corr.ColorBalance,
    corr.QuadrilateralNoBars,
    corr.PerspectiveNoBars,
    corr.SingleFrequencyGreyscale,
    corr.CocentricSineWaves,
    corr.PlasmaNoise,
    corr.VoronoiNoise,
    corr.CausticNoise,
    corr.PerlinNoise,
    corr.BlueNoise,
    corr.BrownishNoise,
    corr.Scatter,
    corr.ChromaticAbberation,
    corr.TransverseChromaticAbberation,
    corr.CircularMotionBlur,
    corr.BleachBypass,
    corr.Technicolor,
    corr.Pseudocolor,
    corr.HueShift,
    corr.ColorDither,
    corr.CheckerBoardCutOut,
    corr.Sparkles,
    corr.InverseSparkles,
    corr.Lines,
    corr.BlueNoiseSample,
    corr.PinchAndTwirl,
    corr.CausticRefraction,
    corr.FishEye,
    corr.WaterDrop,
    corr.Ripple,
]
transform_dict = {t.name : t for t in transform_list}

def build_transform(name, severity, dataset_type):
    assert dataset_type in ['cifar', 'imagenet'],\
            "Only cifar and imagenet image resolutions are supported."
    return transform_dict[name](severity=severity, 
            im_size=(32 if dataset_type=='cifar' else 224)
            )

def build_transforms_from_string(string, dataset_type):
    im_size = (32 if dataset_type=='cifar' else 224)
    transforms = []
    for s in string.split("--"):
        if not s:
            continue
        name, sev = s.split("-")
        t = transform_dict[name]
        transforms.append(t(float(sev),im_size))
    return transforms

def transform_string(transforms):
    string = ''
    for t in transforms:
        if string != '':
            string += "--"
        string = string  + t.name + "-" + str(t.severity)
    if string == '':
        string = '--'
    return string

def get_transforms_by_tag(inclusions, exclusions=[]):
    transforms = []
    for t in transform_list:
        if any([i in t.tags for i in inclusions])\
                and not any([e in t.tags for e in exclusions]):
            transforms.append(t)
    return transforms
