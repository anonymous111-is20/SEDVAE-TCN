import torch
import torch.nn as nn

import numpy as np 
from torchsummary import summary


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:,:,:-self.chomp_size].contiguous()


#padding=(kernel_size-1) * dilation_size
class DepthWiseSeparebleConv1d(nn.Module):
    
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
                causal=True):
        padding = (kernel_size-1)*dilation
        super(DepthWiseSeparebleConv1d, self).__init__()
        if causal:
            self.padding = (kernel_size - 1)*dilation
            self.padding_layer = nn.ConstantPad1d(self.padding, 0)
        else:
            self.padding = (kernel_size*dilation) // 2
            self.padding_layer = nn.ConstantPad1d(self.padding, self.padding)
        self.depthwise = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation,groups=in_channels)
        self.chomp = Chomp1d(padding)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.padding_layer(x)
        out = self.depthwise(out)
        out = self.chomp(out)
        out = self.pointwise(out)
        return out

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, kernel, dilation, padding=[1,1,1], causal=True):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels*2, kernel[0], dilation=dilation[0]),
            nn.PReLU(),
            nn.BatchNorm1d(in_channels*2),
            DepthWiseSeparebleConv1d(in_channels*2, in_channels*2, kernel[1], dilation=dilation[1], causal=causal),
            # Chomp1d((kernel[1] - 1)*dilation),
            nn.PReLU(),
            nn.BatchNorm1d(in_channels*2),
            nn.Conv1d(in_channels*2, in_channels, kernel[2], dilation=dilation[2])
        )

    def forward(self, x):
        h = x
        shape = x.shape[-1]
        # print(x.shape)
        # print("size of input:", h.shape)
        out = self.block(x)[:,:,:shape]
        # out = self.block(x)
        # print("size of output:", out.shape)
        out = out.add_(h)
        
        return out

########################################## 2D convolution######################################
class Chomp2d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:,:,:-self.chomp_size].contiguous()


#padding=(kernel_size-1) * dilation_size
class DepthWiseSeparebleConv2d(nn.Module):
    
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
                causal=True):
        padding = (kernel_size-1)*dilation
        super(DepthWiseSeparebleConv2d, self).__init__()
        if causal:
            self.padding = (kernel_size - 1)*dilation
            self.padding_layer = nn.ConstantPad2d((1,1,self.padding, 0), 0)
        else:
            self.padding = (kernel_size*dilation) // 2
            self.padding_layer = nn.ConstantPad2d((1,1,self.padding, self.padding), 0)
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=(1,1), padding=padding, dilation=dilation,groups=in_channels)
        self.chomp = Chomp2d(padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), padding=0)

    def forward(self, x):
        out = self.padding_layer(x)
        out = self.depthwise(out)
        out = self.chomp(out)
        out = self.pointwise(out)
        return out

class ResidualBlock2(nn.Module):

    def __init__(self, in_channels, kernel, dilation, padding=[1,1,1], causal=True):
        super(ResidualBlock2, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=(kernel[0], kernel[0]), dilation=dilation[0]),
            nn.PReLU(),
            nn.BatchNorm2d(in_channels*2),
            DepthWiseSeparebleConv2d(in_channels*2, in_channels*2, kernel_size=kernel[1], dilation=dilation[1], causal=causal),
            # Chomp1d((kernel[1] - 1)*dilation),
            nn.PReLU(),
            nn.BatchNorm2d(in_channels*2),
            nn.Conv2d(in_channels*2, in_channels, kernel_size=(kernel[2], kernel[2]), dilation=dilation[2])
        )

    def forward(self, x):
        h = x
        shape1 = x.shape[-1]
        shape2 = x.shape[-2]
        # print(x.shape)
        # print("size of input:", h.shape)
        out = self.block(x)[:,:,:shape2,:shape1]
        # out = self.block(x)
        # print("size of output:", out.shape)
        out = out.add_(h)
        
        return out


# model = nn.Sequential(
#     ResidualBlock(1,[1,3,1], [1,1,1]),
#     ResidualBlock(1,[1,3,1], [1,2,1]),
#     ResidualBlock(1,[1,3,1], [1,4,1]),
# ).cuda()
# print(summary(model, (1,256)))
# data = torch.randn((20,1,256))
# result = model(data)
# print(result)
# print(result.shape)