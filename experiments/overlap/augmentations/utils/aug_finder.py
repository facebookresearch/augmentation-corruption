# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ... import augmentations as aug

master_aug_list = [
    aug.pil.AutoContrast,
    aug.pil.Equalize,
    aug.pil.Posterize,
    aug.pil.Solarize,
    aug.pil.Affine,
    aug.pil.ShearX,
    aug.pil.ShearY,
    aug.pil.TranslateX,
    aug.pil.TranslateY,
    aug.pil.Rotate,
    aug.pil.ScaleX,
    aug.pil.ScaleY,
    aug.pil.ScaleFixedAspectRatio,
    aug.pil.Invert,
    aug.pil.ColorBalance,
    aug.pil.Sharpness,
    aug.pil.Contrast,
    aug.pil.Brightness,
    aug.pil.Quadrilateral,
    aug.pil.KeystoneH,
    aug.pil.KeystoneV,
    aug.pil.Perspective,
    aug.pil.QuadrilateralNoBars,
    aug.pil.PerspectiveNoBars,
    aug.additive_noise.SingleFrequencyGreyscale,
    aug.additive_noise.SingleFrequencyColor,
    aug.additive_noise.CocentricSineWaves,
    aug.additive_noise.PlasmaNoise,
    aug.additive_noise.VoronoiNoise,
    aug.additive_noise.CausticNoise,
    aug.additive_noise.PerlinNoise,
    aug.additive_noise.BlueNoise,
    aug.additive_noise.BrownishNoise,
    aug.blurs.Scatter,
    aug.blurs.ChromaticAbberation,
    aug.blurs.TransverseChromaticAbberation,
    aug.blurs.HomogeneousColorBlur,
    aug.blurs.Erosion,
    aug.blurs.Dilation,
    aug.blurs.CircularMotionBlur,
    aug.color.BleachBypass,
    aug.color.Technicolor,
    aug.color.Pseudocolor,
    aug.color.HueShift,
    aug.color.ColorDither,
    aug.obscure.CutOut,
    aug.obscure.CheckerBoardCutOut,
    aug.additive_noise.Sparkles,
    aug.additive_noise.InverseSparkles,
    aug.obscure.Lines,
    aug.obscure.RandomSample,
    aug.obscure.BlueNoiseSample,
    aug.distortion.PinchAndTwirl,
    aug.distortion.PinchAndTwirlV2,
    aug.distortion.CausticRefraction,
    aug.distortion.FishEyeV2,
    aug.distortion.WaterDrop,
    aug.distortion.Ripple,
    aug.imagenetc.GaussianNoise,
    aug.imagenetc.ShotNoise,
    aug.imagenetc.ImpulseNoise,
    aug.imagenetc.SpeckleNoise,
    aug.imagenetc.MotionBlur,
    aug.imagenetc.DefocusBlur,
    aug.imagenetc.ZoomBlur,
    aug.imagenetc.GlassBlur,
    aug.imagenetc.GaussianBlur,
    aug.imagenetc.Brightness,
    aug.imagenetc.Fog,
    aug.imagenetc.Frost,
    aug.imagenetc.Snow,
    aug.imagenetc.Spatter,
    aug.imagenetc.Contrast,
    aug.imagenetc.Pixelate,
    aug.imagenetc.JPEGCompression,
    aug.imagenetc.ElasticTransform,
    aug.imagenetc.Saturate,
]
aug_dict = {a.name : a for a in master_aug_list}

def get_aug_by_name(name):
    return aug_dict[name]

def get_augs_by_tag(inclusions, exclusions=[]):
    augs = []
    for a in master_aug_list:
        skip = False
        for e in exclusions:
            if e in a.tags:
                skip = True
        if skip:
            continue
        include = False
        for i in inclusions:
            if i in a.tags:
                include = True
                break
        if include:
            augs.append(a)
    return augs


def parse_aug_string(aug_string, im_size, max_intensity=False, record=False, **aug_kwargs):
    augs = []
    for s in aug_string.split("--"):
        if not s:
            continue
        name, sev = s.split("-")
        a = aug_dict[name]
        augs.append(a(float(sev),im_size, max_intensity=max_intensity, **aug_kwargs))
    return augs

def build_aug_string(augs):
    aug_string = ''
    for aug in augs:
        if aug_string != '':
            aug_string += "--"
        aug_string = aug_string  + aug.name + "-" + str(aug.severity)
    if aug_string == '':
        aug_string = '--'
    return aug_string
