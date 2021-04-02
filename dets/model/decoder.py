

import torch
import torch.nn as nn


from .module import conv_bn_relu


class YOLOFDecoder(nn.Module):
    '''
    '''
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()

        cls_decoder = nn.Sequential(*[conv_bn_relu(in_channels, in_channels, 3, 1, 1) for _ in range(2)])
        self.cls_decoder = nn.ModuleList([
            cls_decoder,
            # conv_bn_relu(in_channels, num_anchors * num_classes, 1, 1, 0)
            nn.Conv2d(in_channels, num_anchors * num_classes, 1, 1, 0)
        ])
        
        reg_decoder = nn.Sequential(*[conv_bn_relu(in_channels, in_channels, 3, 1, 1) for _ in range(4)])
        self.reg_decoder = nn.ModuleList([
            reg_decoder,
            # conv_bn_relu(in_channels, num_anchors * 4, 1, 1, 0)
            nn.Conv2d(in_channels, num_anchors * 4, 1, 1, 0)
        ])


    def forward(self, x):
        '''
        '''
        _cls = self.cls_decoder[0](x)
        _reg = self.reg_decoder[0](x)

        _cls = self.cls_decoder[1](_cls * _reg)
        _reg = self.reg_decoder[1](_reg)

        return _cls, _reg



class YOLOV3Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        num_classes = 80
        num_anchors = 3
        num_layers = 3

        in_channels = 256
        out_channels = (4 + 1 + num_classes) * num_anchors

        block = lambda c_in, c_out: nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, 1, 1),
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, c_out, 1, 1, 0),
        )

        self.model = nn.ModuleList([block(in_channels, out_channels) for _ in range(num_layers)])
        
        self._init_parameters()


    def _init_parameters(self, ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)


    def forward(self, feats):
        '''
        Args:
            feats (tensor or Sequential[tensor]): feats come from backbone
        Returns:
            preds (tensor or Sequential[Tensor]): decode feats for detection
        '''
        if not isinstance(feats, (tuple, list)):
            feats = (feats, )
        
        outputs = []
        for feat, m in zip(feats, self.model):
            outputs.append(m(feat))
        
        return outputs

