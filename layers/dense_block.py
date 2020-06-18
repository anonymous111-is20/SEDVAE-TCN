import math 
import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import numpy as np
import sys

class TorchOLA(nn.Module):
    r"""Overlap and add on gpu using torch tensor"""
    # Expects signal at last dimension
    def __init__(self, frame_shift=256):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift
    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device, requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels*r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r
    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels//self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels//self.r, H, -1))
        return out

class SPConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose1d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels*r, kernel_size=kernel_size, stride=1)
        self.r = r
    def forward(self, x):
        out = self.conv(x)
        batch_size, channels, length = out.shape
        out = out.view((batch_size, self.r,channels//self.r, length))
        out = out.permute(0,2,3,1)
        out = out.contiguous().view((batch_size, channels//self.r, -1))
        return out
        
class DenseBlock(nn.Module):
    def __init__(self, input_size, depth=5, in_channels=64, causal=True):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        self.causal = causal
        for i in range(self.depth):
            dil = 2**i
            if causal:
                pad_length = self.twidth + (dil-1)*(self.twidth-1)-1
                setattr(self, 'pad{}'.format(i+1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            else:
                pad_length = (self.twidth*dil) // 2
                setattr(self, 'pad{}'.format(i+1), nn.ConstantPad2d((1, 1, pad_length, pad_length), value=0.))
            setattr(self, 'conv{}'.format(i+1),
                    nn.Conv2d(self.in_channels*(i+1), self.in_channels, kernel_size=self.kernel_size, dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i+1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i+1), nn.PReLU(self.in_channels))
    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i+1))(skip)
            out = getattr(self, 'conv{}'.format(i+1))(out)
            out = getattr(self, 'norm{}'.format(i+1))(out)
            out = getattr(self, 'prelu{}'.format(i+1))(out)[:,:,:100,:]
            # print('out: ', out.shape)
            # print('skip: ', skip.shape)
            skip = torch.cat([out, skip], dim=1)
        return out
        