

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
            nn.Conv2d(in_channels, num_anchors * num_classes, 1, 1, 0)
        ])
        
        reg_decoder = nn.Sequential(*[conv_bn_relu(in_channels, in_channels, 3, 1, 1) for _ in range(4)])
        self.reg_decoder = nn.ModuleList([
            reg_decoder,
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
    '''
    c -> (4-box + 1-obj + num_cls) * anchor
    '''
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



class RetinaDecoder(nn.Module):
    '''two branch
    cls branch
    box branch
    '''
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, feats):
        pass



class SSDDecoder(nn.Module):
    '''
    '''
    def __init__(self, cfg=None):
        super().__init__()

        num_classes = 80
        num_anchors = [5, 5, 5]
        num_layers = 3

        in_channels = 256
        out_channels_list = [(4 + 1 + num_classes) * a for a in num_anchors]

        block = lambda c_in, c_out: nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, 1, 1),
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, c_in, 3, 1, 1),
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            
            nn.Conv2d(c_in, c_out, 1, 1, 0),
        )
        self.model = nn.ModuleList([block(in_channels, o) for o in out_channels_list])


    def forward(self, feats):
        if not isinstance(feats, (tuple, list)):
            feats = (feats, )

        outputs = []
        for m, feat in zip(self.model, feats):
            outputs.append(m(feat))
        
        return outputs

        



class DETRDecoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        in_channels = 2048
        hidden_dim = 256
        num_classes = 80
        num_query = 100

        self.conv = nn.Conv2d(in_channels, hidden_dim, 1, 1)
        self.transformer = nn.Transformer(hidden_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        self.num_query = num_query
        self.query_pos = nn.Parameter(torch.rand(num_query, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim//2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim//2))
        # self.row_embed = nn.Embedding(50, hidden_dim//2)
        # self.col_embed = nn.Embedding(50, hidden_dim//2)


    def forward(self, feats):
        '''
        '''
        if not isinstance(feats, (list, tuple)):
            feats = (feats, )

        assert len(feats) == 1, ''

        hidden = self.conv(feats[0])
        _n, _c, _h, _w = hidden.shape

        pos = torch.cat([
            self.col_embed[:_w].unsqueeze(0).repeat(_h, 1, 1), 
            self.row_embed[:_h].unsqueeze(1).repeat(1, _w, 1)], 
            dim=-1).view(_h * _w, 1, _c) # flatten(0, 1).unsqueeze(1)

        # n c h w -> l<h * w> n c
        hidden = hidden.view(_n, _c, -1).permute(2, 0, 1).contiguous()

        # l_scr<_h * _w> -> l_trg<100> 
        hidden = self.transformer(pos + hidden, self.query_pos.unsqueeze(1).repeat(1, _n, 1))

        outputs = {}
        outputs['pred_logits'] = self.linear_class(hidden).permute(1, 0, 2).contiguous()
        outputs['pred_boxes'] = self.linear_bbox(hidden).sigmoid().permute(1, 0, 2).contiguous()

        return outputs


    