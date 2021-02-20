# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResHead(nn.Module):
    """ResNet head."""

    def __init__(self, w_in, nc):
        super(ResHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        self.features = x
        x = self.fc(x)
        return x

class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, w_b, num_gs,
            bn_params, stride_1x1, relu_inplace):
        super(BottleneckTransform, self).__init__()
        self._construct(w_in, w_out, stride, w_b, num_gs,
                bn_params, stride_1x1, relu_inplace)

    def _construct(self, w_in, w_out, stride, w_b, num_gs,
            bn_params, stride_1x1, relu_inplace):
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        (str1x1, str3x3) = (stride, 1) if stride_1x1 else (1, stride)
        # 1x1, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_b, kernel_size=1,
            stride=str1x1, padding=0, bias=False
        )
        self.a_bn = torch.nn.BatchNorm2d(w_b, **bn_params)

        self.a_relu = nn.ReLU(inplace=relu_inplace)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            w_b, w_b, kernel_size=3,
            stride=str3x3, padding=1, groups=num_gs, bias=False
        )
        self.b_bn = torch.nn.BatchNorm2d(w_b, **bn_params)
        self.b_relu = nn.ReLU(inplace=relu_inplace)
        # 1x1, BN
        self.c = nn.Conv2d(
            w_b, w_out, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.c_bn = torch.nn.BatchNorm2d(w_out, **bn_params)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResBlock(nn.Module):
    """Residual block: x + F(x)"""

    def __init__(
        self, w_in, w_out, stride, w_b, num_gs, bn_params, stride_1x1, relu_inplace
    ):
        super(ResBlock, self).__init__()
        self._construct(w_in, w_out, stride, w_b, num_gs,
                bn_params, stride_1x1, relu_inplace)

    def _add_skip_proj(self, w_in, w_out, stride, bn_params):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1,
            stride=stride, padding=0, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(w_out, **bn_params)

    def _construct(self, w_in, w_out, stride, w_b, num_gs, 
            bn_params, stride_1x1, relu_inplace):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride, bn_params)
        self.f = BottleneckTransform(w_in, w_out, stride, w_b, num_gs,
                bn_params, stride_1x1, relu_inplace)
        self.relu = nn.ReLU(relu_inplace)

    def forward(self, x):
        fx = self.f(x)
        if self.proj_block:
            x = self.bn(self.proj(x))
        x = x + fx
        x = self.relu(x)
        return x


class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b, num_gs,
            bn_params, stride_1x1, relu_inplace):
        super(ResStage, self).__init__()
        self._construct(w_in, w_out, stride, d, w_b, num_gs,
                bn_params, stride_1x1, relu_inplace)

    def _construct(self, w_in, w_out, stride, d, w_b, num_gs, bn_params, stride_1x1, relu_inplace):
        # Construct the blocks
        for i in range(d):
            # Stride and w_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Construct the block
            res_block = ResBlock(
                b_w_in, w_out, b_stride, w_b, num_gs,
                bn_params, stride_1x1, relu_inplace
            )
            self.add_module('b{}'.format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class ResStem(nn.Module):
    """Stem of ResNet."""

    def __init__(self, w_in, w_out, bn_params, relu_inplace):
        super(ResStem, self).__init__()
        self._construct_imagenet(w_in, w_out, bn_params, relu_inplace)

    def _construct_imagenet(self, w_in, w_out, bn_params, relu_inplace):
        # 7x7, BN, ReLU, maxpool
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(w_out, **bn_params)
        self.relu = nn.ReLU(relu_inplace)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResNetPycls(nn.Module):
    """ResNeXt model."""

    def __init__(self, depth=50, width_factor=1, num_groups=1, width_per_group=64, 
            num_classes=1000, bn_params={'eps':1e-5, 'momentum':0.1, 'affine':True},
            stride_1x1=False, relu_inplace=True, final_gamma=True
            ):
        super(ResNetPycls, self).__init__()
        self.depth = depth
        self.width = width_factor
        self.ng = num_groups
        self.width_per_group = width_per_group
        self.num_classes = num_classes
        self.bn_params = bn_params
        self.stride_1x1 = stride_1x1
        self.relu_inplace = relu_inplace
        self._construct_imagenet()

        def init_weights(m, cfg):
            """Performs ResNet-style weight initialization."""
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm2d):
                zero_init_gamma = (
                    hasattr(m, 'final_bn') and m.final_bn and
                    final_gamma
                )
                m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

        self.apply(lambda m : init_weights(m, final_gamma))


    def _construct_imagenet(self):
        # Retrieve the number of blocks per stage
        (d1, d2, d3, d4) = _IN_STAGE_DS[self.depth]
        # Compute the initial bottleneck width
        num_gs = self.ng
        w_b = self.width_per_group * num_gs
        w1, w2, w3, w4 = [self.width * w for w in [256, 512, 1024, 2048]]
        # Stem: (N, 3, 224, 224) -> (N, 64, 56, 56)
        self.stem = ResStem(w_in=3, w_out=64, bn_params=self.bn_params, relu_inplace=self.relu_inplace)
        # Stage 1: (N, 64, 56, 56) -> (N, 256, 56, 56)
        self.s1 = ResStage(
            w_in=64, w_out=w1, stride=1, d=d1,
            w_b=w_b, num_gs=num_gs, 
            bn_params=self.bn_params, stride_1x1=self.stride_1x1, relu_inplace=self.relu_inplace
        )
        # Stage 2: (N, 256, 56, 56) -> (N, 512, 28, 28)
        self.s2 = ResStage(
            w_in=w1, w_out=w2, stride=2, d=d2,
            w_b=w_b * 2, num_gs=num_gs,
            bn_params=self.bn_params, stride_1x1=self.stride_1x1, relu_inplace=self.relu_inplace
        )
        # Stage 3: (N, 512, 56, 56) -> (N, 1024, 14, 14)
        self.s3 = ResStage(
            w_in=w2, w_out=w3, stride=2, d=d3,
            w_b=w_b * 4, num_gs=num_gs,
            bn_params=self.bn_params, stride_1x1=self.stride_1x1, relu_inplace=self.relu_inplace
        )
        # Stage 4: (N, 1024, 14, 14) -> (N, 2048, 7, 7)
        self.s4 = ResStage(
            w_in=w3, w_out=w4, stride=2, d=d4,
            w_b=w_b * 8, num_gs=num_gs,
            bn_params=self.bn_params, stride_1x1=self.stride_1x1, relu_inplace=self.relu_inplace
        )
        # Head: (N, 2048, 7, 7) -> (N, num_classes)
        self.head = ResHead(w_in=w4, nc=self.num_classes)


    def forward(self, x):
        for module in self.children():
            x = module(x)
            if isinstance(module, ResHead):
                self.features = module.features
        return x

