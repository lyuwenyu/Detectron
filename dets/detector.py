
import torch
import torch.nn as nn
from torch import Tensor

from .model import backbone, encoder, decoder, target



class YOLOV3Detector(nn.Module):
    '''
    '''
    def __init__(self, cfg=None):
        super().__init__()
        
        _backbone = backbone.Resnet50(pretrained=True)
        _encoder = encoder.FPNEncoder(_backbone.out_channels_list, 256)
        _decoder = decoder.YOLOV3Decoder(cfg)
        _target = target.YOLOV3Target(cfg)

        self.model = nn.ModuleList([_backbone, _encoder, _decoder, _target])

    def forward(self, data, targets=None):
        '''
        '''
        for i in range(len(self.model) - 1):
            data = self.model[i](data)

        output = self.model[-1](data, targets)

        return output



class YOLOFDetector(nn.Module):
    '''
    '''
    def __init__(self, cfg=None):
        super().__init__()
        pass

    def forward(self, data, targets=None):
        pass


class DETRDetector(nn.Module):
    '''
    '''
    pass




if __name__ == '__main__':

    yolov3 = YOLOV3Detector()