
import torch
import torch.nn as nn  
import torch.nn.functional as F

import math

from collections import defaultdict

from ..utils.bbox import wh_iou, bbox_iou


class YOLOV3Target(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

        self.cfg = cfg

        self.iou_threshold = 0.2  # 0.2
        self.ignore_threshold = 0.5
        self.wh_threshold = 4 # math.e ** 2
        self.wh_mode = 'v3' # 'sigmoid', 'exp'

        self.num_anchors = 3
        self.num_classes = 80
        self.num_outputs = 4 + 1 + self.num_classes 

        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        self.anchors = torch.tensor(anchors).view(-1, self.num_anchors, 2).to(torch.float32)
        self.strides = [8, 16, 32]


    def forward(self, preds, targets=None):
        '''
        Args:
            preds (tensor or tuple(Tensor)): [N, C, H, W]
            targets (tensor): M * 6, where 6 is [im-idx, cls, x, y, w, h]
        Returns:
            Tensor [n, -1, c]: c ==> [bbox, obj, cls]
            loss: when targets is not None
        '''
        if not isinstance(preds, (tuple, list)):
            preds = (preds, )
        
        outputs = []
        losses = defaultdict(int)

        for i, pred in enumerate(preds):

            n, _, h, w = pred.shape
            pred = pred.view(n, self.num_anchors, -1, h, w).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training or targets is None:
                # grid
                grids = self._make_grid(h, w, pred.device, pred.dtype)
                anchors = self.anchors[i].to(pred.device, pred.dtype)

                # x y
                pred[..., 0:2] = (pred[..., 0:2].sigmoid() + grids.view(1, 1, h, w, 2)) * self.strides[i]

                # w h
                if self.wh_mode == 'v3':
                    pred[..., 2:4] = pred[..., 2:4].exp() * anchors.view(1, -1, 1, 1, 2)
                elif self.wh_mode == 'v5':
                    # pred[..., 2:4] = math.e ** (2 * (pred[..., 2:4].sigmoid() - 0.5)) ** 2 * anchors.view(1, -1, 1, 1, 2)
                    pred[..., 2:4] = (2 * pred[..., 2:4].sigmoid()).pow(2) * anchors.view(1, -1, 1, 1, 2)

                # obj cls
                pred[..., 4:].sigmoid_()

                pred = pred.view(n, -1, self.num_outputs)
                outputs.append(pred)

            else:
                _targets = targets.clone()
                _targets[:, [2, 4]] *= w
                _targets[:, [3, 5]] *= h 
                _anchors = self.anchors[i].to(pred.device, pred.dtype) / self.strides[i]

                loss, lbox, lobj, lcls = self._compute_loss(pred, _anchors, _targets)
                losses['loss'] += loss
                losses['lcls'] += lcls.item()
                losses['lbox'] += lbox.item()
                losses['lobj'] += lobj.item()

        return torch.cat(outputs, dim=1) if (not self.training or targets is None) else losses


    @staticmethod
    def _make_grid(h, w, device, dtype=torch.float,):
        grid_h, grid_w = torch.meshgrid([torch.arange(h), torch.arange(w)])
        # grid = torch.cat((grid_h.unsqueeze(-1), grid_w.unsqueeze(-1)), dim=-1).to(dtype=pred.dtype, device=pred.device)
        grid = torch.stack((grid_w, grid_h), 2).view((1, 1, h, w, 2)).to(dtype=dtype, device=device)
        return grid


    def _compute_loss(self, pred, anchors, targets):
        '''
        '''
        lcls, lbox, lobj = [torch.zeros(1, device=pred.device) for _ in range(3)]
        
        tobj = torch.zeros_like(pred[..., 0], device=pred.device)
        (im_idx, an_idx, j_idx, i_idx, mask), t_cls, t_xy, t_wh, t_box = self._build_target(pred.detach(), anchors, targets)

        if t_cls.shape[0] > 0:

            p = pred[im_idx, an_idx, j_idx, i_idx]

            # t_obj
            tobj[im_idx, an_idx, j_idx, i_idx] = 1.

            # # p_bbox
            _p_xy = p[:, 0:2].sigmoid()
            if self.wh_mode == 'v3':
                _p_wh = p[..., 2:4].exp() * anchors[an_idx]
            elif self.wh_mode == 'v5':
                # _p_wh = math.e ** (2 * (p[:, 2:4].sigmoid() - 0.5)) ** 2 * anchors[an_idx]
                _p_wh = (2 * p[:, 2:4].sigmoid()).pow(2) * anchors[an_idx]

            p_box = torch.cat((_p_xy, _p_wh), dim=-1)
            iou = bbox_iou(p_box.T, t_box, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()

            # p_xy = p[:, 0:2]
            lbox += F.binary_cross_entropy_with_logits(p[:, 0:2], t_xy, reduction='mean')

            # p_wh = p[:, 2:4]
            if self.wh_mode == 'v3':
                lbox += F.smooth_l1_loss(p[:, 2:4], t_wh, reduction='mean')
            elif self.wh_mode == 'v5':
                lbox += F.binary_cross_entropy_with_logits(p[:, 2:4], t_wh, reduction='mean')

            # p_cls
            if self.num_classes > 1:
                _cls = torch.eye(self.num_classes, device=pred.device, dtype=pred.dtype)[t_cls]
                lcls += F.binary_cross_entropy_with_logits(p[:, 5:], _cls, reduction='mean')
        
        # p_obj
        lobj += F.binary_cross_entropy_with_logits(pred[..., 4], tobj, reduction='mean')
        # lobj += F.binary_cross_entropy_with_logits(pred[..., 4][~mask], t_obj[~mask], reduction='mean')
        # lobj += F.binary_cross_entropy_with_logits(pred[..., 4][t_obj==1], t_obj[t_obj==1], reduction='mean')
        # lobj += F.binary_cross_entropy_with_logits(pred[..., 4][t_obj==0], t_obj[t_obj==0], reduction='mean')
        # lobj += F.binary_cross_entropy_with_logits(pred[..., 4][(t_obj==0)&~mask], t_obj[(t_obj==0)&~mask], reduction='mean')

        loss = 0.5 * lcls + 0.05 * lbox + lobj

        return loss, 0.05 * lbox, lobj, 0.5 * lcls


    def _build_target(self, pred, anchors, targets):
        '''
        target [nt, [im-idx, cls, x, y, w, h]]
        '''
        n = targets.shape[0]
        a = torch.arange(self.num_anchors, device=targets.device).float().view(-1, 1).repeat(1, n)
        t = torch.cat((targets.repeat(self.num_anchors, 1, 1), a[:, :, None]), 2) # 3 * nt * (6 + 1)

        if n:
            if self.wh_mode == 'v5':
                r = t[:, :, 4:6] / anchors[:, None, :]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.wh_threshold
                t = t[j] # M * 7

            if t.shape[0]:
                t = t.view(-1, 7)
                ious = wh_iou(anchors, t[:, 4:6])
                _value, _index = ious.max(0)
                j = (_index == t[:, -1]) & (_value > self.iou_threshold)
                
                ignore = ~j & (_value > self.ignore_threshold)
                ignore = t[ignore]

                t = t[j]

        im_idx = t[:, 0].long()
        an_idx = t[:, -1].long()
        i_idx, j_idx = t[:, 2:4].long().T

        t_cls = t[:, 1].long()
        t_xy = t[:, 2:4] - t[:, 2:4].long().float()
        
        if self.wh_mode == 'v3':
            t_wh = torch.log(t[:, 4:6] / anchors[an_idx])
        elif self.wh_mode == 'v5':
            # t_wh = torch.log( torch.sqrt( t[:, 4:6] / anchors[an_idx] ) ) / 2. + 0.5
            t_wh = torch.sqrt(t[:, 4:6] / anchors[an_idx]) / 2.
        
        t_box = torch.cat((t_xy, t[:, 4:6]), 1)

        mask = torch.zeros_like(pred[..., 4], device=pred.device, dtype=torch.bool)
        mask[ignore[:, 0].long(), ignore[:, -1].long(), ignore[:, 3].long(), ignore[:, 2].long()] = 1

        return (im_idx, an_idx, j_idx, i_idx, mask), t_cls, t_xy, t_wh, t_box



class SSDTarget(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        prior_bbox = []

        self.register_buffer('prior_bbox', prior_bbox)


    def forward(self, feats, targets=None):
        '''
        '''
        if not isinstance(preds, (tuple, list)):
            preds = (preds, )
        
        pass