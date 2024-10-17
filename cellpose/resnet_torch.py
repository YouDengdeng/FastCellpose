import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime

from . import transforms, io, dynamics, utils

sz = 3


def convbatchrelu(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )


def batchconv(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


# class resup(nn.Module):
#     def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):  # 32,16,128,3,False
#         super().__init__()
#         self.conv = nn.Sequential()
#         self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
#         self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz,
#                                                       concatenation=concatenation))
#         self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz))
#         self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz))
#         self.proj = batchconv0(in_channels, out_channels, 1)
#
#     def forward(self, x, y, style, mkldnn=False):
#         x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
#         x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
#         return x
#
#
# class resdown(nn.Module):
#     def __init__(self, in_channels, out_channels, sz):
#         super().__init__()
#         self.conv = nn.Sequential()
#         self.proj = batchconv0(in_channels, out_channels, 1)
#         for t in range(4):
#             if t == 0:
#                 self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
#             else:
#                 self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))
#
#     def forward(self, x):
#         x = self.proj(x) + self.conv[1](self.conv[0](x))
#         x = x + self.conv[3](self.conv[2](x))
#         return x


class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, concatenation=concatenation))
        self.proj = batchconv0(in_channels, out_channels, 1)

    def forward(self, x, y, style, mkldnn=False):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        return x


class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1)
        for t in range(2):
            if t == 0:
                self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))

    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        return x

#
# class convup(nn.Module):  # convup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation)
#     def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
#         super().__init__()
#         self.conv = nn.Sequential()
#         self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
#         self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz,
#                                                       concatenation=concatenation))
#         self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz))
#         self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz))
#
#     def forward(self, x, y, style, mkldnn=False):
#         x = self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)  # style, x, mkldnn=False, y=None
#         x = self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
#         return x
# class convdown(nn.Module):
#     def __init__(self, in_channels, out_channels, sz):
#         super().__init__()
#         self.conv = nn.Sequential()
#         for t in range(4):
#             if t == 0:
#                 self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
#             else:
#                 self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))
#     def forward(self, x):
#         x = self.conv[0](x)
#         x = self.conv[1](x)
#         x = self.conv[2](x)
#         x = self.conv[3](x)
#         return x


# class convup(nn.Module):
#     def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
#         super().__init__()
#         self.conv = nn.Sequential()
#         self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
#         self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz,
#                                                       concatenation=concatenation))
#
#     def forward(self, x, y, style, mkldnn=False):
#         x = self.conv[1](style, self.conv[0](x), y=y)
#         return x
# class convdown(nn.Module):
#     def __init__(self, in_channels, out_channels, sz):
#         super().__init__()
#         self.conv = nn.Sequential()
#         for t in range(2):
#             if t == 0:
#                 self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
#             else:
#                 self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))
#
#     def forward(self, x):
#         x = self.conv[0](x)
#         x = self.conv[1](x)
#         return x

class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False, need2upsample=True):
        super().__init__()
        self.conv = nn.Sequential()
        self.need2up = need2upsample
        if need2upsample:
            self.conv.add_module('conv_0', nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0))
            self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz,
                                                          concatenation=concatenation))
        else:
            self.conv.add_module('conv_0', batchconvstyle(in_channels, out_channels, style_channels, sz,
                                                          concatenation=concatenation))

    def forward(self, x, y, style, mkldnn=False):
        if self.need2up:
            x = self.conv[1](style, self.conv[0](x), y=y)
        else:
            x = self.conv[0](style, x, y=y)
        return x


class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t == 0:
                self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))

    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x


class downsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase) - 1):
            if residual_on:
                self.down.add_module('res_down_%d' % n, resdown(nbase[n], nbase[n + 1], sz))
            else:
                self.down.add_module('conv_down_%d' % n, convdown(nbase[n], nbase[n + 1], sz))

    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd


class batchconvstyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.concatenation = concatenation
        if concatenation:
            self.conv = batchconv(in_channels * 2, out_channels, sz)
            self.full = nn.Linear(style_channels, out_channels * 2)
        else:
            self.conv = batchconv(in_channels, out_channels, sz)
            self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False, y=None):
        if y is not None:
            if self.concatenation:
                x = torch.cat((y, x), dim=1)
            else:
                x = x + y
        feat = self.full(style)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat.unsqueeze(-1).unsqueeze(-1)).to_mkldnn()
        else:
            y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y


class make_style(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0):
        # style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style ** 2, axis=1, keepdim=True) ** .5

        return style


class upsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True, concatenation=False):  # nbase=[16,32,64,128,128]
        super().__init__()
        # self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            need2upsample = True
            if n == len(nbase) - 1:
                need2upsample = False
            if residual_on:
                self.up.add_module('res_up_%d' % (n - 1),
                                   resup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation))  # 32,16,128,3,False
            else:
                self.up.add_module('conv_up_%d' % (n - 1),
                                   convup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation,
                                          need2upsample=need2upsample))
                # nn.ConvTranspose2d(nbase[n], nbase[n], sz, stride=2, padding=0,
                #                    output_padding=0, groups=1,
                #                    bias=True, dilation=1)

    def forward(self, style, xd, mkldnn=False):
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up) - 2, -1, -1):
            # if mkldnn:
            #     x = self.upsampling(x.to_dense()).to_mkldnn()
            # else:
            #     x = self.upsampling(x)
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x


import torch.nn.init as init


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


class CPnet(nn.Module):
    def __init__(self, nbase, nout, sz,
                 residual_on=True, style_on=True,
                 concatenation=False, mkldnn=False,
                 diam_mean=30.):
        super(CPnet, self).__init__()
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, sz, residual_on=residual_on)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, residual_on=residual_on, concatenation=concatenation)
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], nout, 1)
        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.style_on = style_on
        self.apply(weights_init)

    def forward(self, data):
        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)
        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense())
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T0 = self.upsample(style, T0, self.mkldnn)
        T0 = self.output(T0)
        if self.mkldnn:
            T0 = T0.to_dense()
            # T1 = T1.to_dense()
        return T0, style0
        # return T0

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        if not cpu:
            state_dict = torch.load(filename)
        else:
            self.__init__(self.nbase,
                          self.nout,
                          self.sz,
                          self.residual_on,
                          self.style_on,
                          self.concatenation,
                          self.mkldnn,
                          self.diam_mean)
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
        self.load_state_dict(dict([(name, param) for name, param in state_dict.items()]), strict=False)
