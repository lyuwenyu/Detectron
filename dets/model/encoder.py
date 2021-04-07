
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import conv_bn, conv_bn_relu


from typing import Sequence, Union, List


def dilataed_encoder_block(in_channels, dilation=1):
    '''
    '''
    assert in_channels % 4 == 0, ''
    return nn.Sequential(
        conv_bn_relu(in_channels, in_channels//4, 1, 1),
        conv_bn_relu(in_channels//4, in_channels//4, 3, 1, dilation, dilation),
        conv_bn_relu(in_channels//4, in_channels, 1, 1)
    )


class DilatedEncoder(nn.Module):
    '''
    '''
    def __init__(self, in_channels=2048, out_channels=512, dilation_rates=(2, 2, 2)):
        super().__init__()

        self.projector = nn.Sequential(
            conv_bn(in_channels, out_channels, 1, 1, 0),
            conv_bn(out_channels, out_channels, 3, 1, 1),
        )

        self.blocks = nn.ModuleList([dilataed_encoder_block(out_channels, rate) for rate in dilation_rates])


    def forward(self, x):
        
        x = self.projector(x)

        for block in self.blocks:
            x += block(x)
        
        return x





class FPNEncoder(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        # in_channels_list = [512, 1024, 2048]
        # out_channels = 256
        num_layers = len(in_channels_list)
        
        block = lambda c_in, c_out, k, s, p: nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )

        # self.inner_blocks = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1) for in_channels in in_channels_list])
        self.inner_blocks = nn.ModuleList([block(in_channels, out_channels, 1, 1, 0) for in_channels in in_channels_list])
        self.layer_blocks = nn.ModuleList([block(out_channels, out_channels, 3, 1, 1) for _ in range(num_layers)])

        self._init_parameters()


    def _init_parameters(self, ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)


    def forward(self, feats):
        '''
        Args:
            feats (tensor or list[tensor]): feats come from backbone
        Returns:
            preds (tensor or list[Tensor]): encode feats for detection
        '''
        if not isinstance(feats, (list, tuple)):
            feats = (feats, )
        
        outputs = []

        last_inner = self.inner_blocks[-1](feats[-1])
        outputs.append(self.layer_blocks[-1](last_inner))

        for i in range(len(feats)-2, -1, -1):
            inner_lateral = self.inner_blocks[i](feats[i])
            inner_top_down = F.interpolate(last_inner, feats[i].shape[-2:], mode='nearest')
            last_inner = inner_lateral + inner_top_down

            outputs.insert(0, self.layer_blocks[i](last_inner))

        return outputs


class DETREncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        in_channels = 2048
        hidden_dim = 256
        num_classes = 80

        self.conv = nn.Conv2d(in_channels, hidden_dim, 1, 1)
        self.transformer = nn.Transformer(hidden_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        self.num_query = 100
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim//2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim//2))


    def forward(self, feats):
        '''
        '''
        if not isinstance(feats, (list, tuple)):
            feats = (feats, )

        assert len(feats) == 1, ''

        hidden = self.conv(feats[0])
        # n c h w -> l<h * w> n c
        _n, _c, _h, _w = hidden.shape
        pos = torch.cat([
            self.col_embed[:_w].unsqueeze(0).repeat(_h, 1, 1), 
            self.row_embed[:_h].unsqueeze(1).repeat(1, _w, 1)], 
            dim=-1).view(_h * _w, 1, _c) # flatten(0, 1).unsqueeze(1)

        # l_scr<_h * _w> -> l_trg<100> 
        hidden = self.transformer(pos + hidden.view(_n, _c, -1).permute(2, 0, 1), self.query_pos.unsqueeze(1).repeat(1, _n, 1))

        return self.linear_class(hidden), self.linear_bbox(hidden).sigmoid()
