
import torch
import torch.nn as nn



def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1):
    '''
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
        nn.BatchNorm2d(out_channels),
    )


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1):
    '''
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )




