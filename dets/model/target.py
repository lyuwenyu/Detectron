
import torch
import torch.nn as nn  
import torch.nn.functional as F

from collections import defaultdict

from ..utils.bbox import wh_iou, bbox_iou


class YOLOV3Target(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

        self.cfg = cfg

        self.iou_threshold = 0.1  # 0.2
        self.wh_threshold = 4.0 
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
            scaled_anchors = self.anchors[i].to(pred.device, pred.dtype) / self.strides[i]

            if not self.training or targets is None:
                grid_h, grid_w = torch.meshgrid([torch.arange(h), torch.arange(w)])
                grid = torch.cat((grid_h.unsqueeze(-1), grid_w.unsqueeze(-1)), dim=-1).to(dtype=pred.dtype, device=pred.device)

                # x y
                pred[..., 0:2] = (pred[..., 0:2].sigmoid() + grid.view(1, 1, h, w, 2)) * self.strides[i]

                # w h
                if self.wh_mode == 'v3':
                    pred[..., 2:4] = pred[..., 2:4].exp() * scaled_anchors.view(1, -1, 1, 1, 2) * self.strides[i]
                elif self.wh_mode == 'v5':
                    pred[..., 2:4] = 2 ** (2 * (pred[..., 2:4].sigmoid() - 0.5)) ** 2 * scaled_anchors.view(1, -1, 1, 1, 2) * self.strides[i]
                    # pred[..., 2:4] = (2 * pred[..., 2:4].sigmoid()).pow(2) * scaled_anchors.view(1, -1, 1, 1, 2) * self.strides[i]

                # obj cls
                pred[..., 4:].sigmoid_()

                pred = pred.view(n, -1, self.num_outputs)
                outputs.append(pred)

            else:
                t = targets.clone()
                t[:, [2, 4]] *= w
                t[:, [3, 5]] *= h 

                loss, lbox, lobj, lcls = self._compute_loss(pred, scaled_anchors, t)
                losses['loss'] += loss
                losses['lcls'] += lcls.item()
                losses['lbox'] += lbox.item()
                losses['lobj'] += lobj.item()

        return torch.cat(outputs, dim=1) if (not self.training or targets is None) else losses


    def _compute_loss(self, pred, anchors, targets):
        '''
        '''
        lcls, lbox, lobj = [torch.zeros(1, device=pred.device) for _ in range(3)]
        
        tobj = torch.zeros_like(pred[..., 0], device=pred.device)
        t, (im_idx, an_idx, j_idx, i_idx), tcls, t_xy, t_wh = self._build_target(pred.detach(), anchors, targets)

        if tcls.shape[0] > 0:

            p = pred[im_idx, an_idx, j_idx, i_idx]

            # t_obj
            tobj[im_idx, an_idx, j_idx, i_idx] = 1.

            # # p_bbox
            # _p_xy = p[:, 0:2].sigmoid() + torch.cat((j_idx.unsqueeze(-1), i_idx.unsqueeze(-1)), dim=-1)
            # _p_wh = 2 ** (2 * (p[:, 2:4].sigmoid() - 0.5)) ** 2 * anchors[an_idx]
            # p_box = torch.cat((_p_xy, _p_wh), dim=1)
            # iou = bbox_iou(p_box.T, t[:, 2:6], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            # lbox += (1.0 - iou).mean()

            # p_xy = p[:, 0:2]
            lbox += F.binary_cross_entropy_with_logits(p[:, 0:2], t_xy, reduction='mean')

            # p_wh = p[:, 2:4]
            if self.wh_mode == 'v3':
                lbox += F.mse_loss(p[:, 2:4], t_wh, reduction='mean')
            elif self.wh_mode == 'v5':
                lbox += F.binary_cross_entropy_with_logits(p[:, 2:4], t_wh, reduction='mean')

            # p_cls
            if self.num_classes > 1:
                _cls = torch.eye(self.num_classes, device=pred.device, dtype=pred.dtype)[tcls]
                lcls += F.binary_cross_entropy_with_logits(p[:, 5:], _cls, reduction='mean')
        
        # p_obj
        lobj += F.binary_cross_entropy_with_logits(pred[..., 4], tobj, reduction='mean')

        loss = 0.5 * lcls + 0.05 * lbox + lobj

        return loss, lbox, lobj, lcls


    def _build_target(self, pred, anchors, targets):
        '''
        target [nt, [im-idx, cls, x, y, w, h]]
        '''
        nt = targets.shape[0]
        ai = torch.arange(self.num_anchors, device=targets.device).float().view(-1, 1).repeat(1, nt)
        t = torch.cat((targets.repeat(self.num_anchors, 1, 1), ai[:, :, None]), 2) # 3 * nt * (6 + 1)

        if nt:
            if self.wh_mode == 'v5':
                r = t[:, :, 4:6] / anchors[:, None, :]  # wh ratio
                j = torch.maximum(r, 1. / r).max(2)[0] < self.wh_threshold
                t = t[j] # M * 7

            t = t.view(-1, 7)
            ious = wh_iou(anchors, t[:, 4:6])
            _value, _index = ious.max(0)
            j = (_index == t[:, -1]) & (_value > self.iou_threshold)
            t = t[j]

        im_idx = t[:, 0].long()
        an_idx = t[:, -1].long()
        i_idx, j_idx = t[:, [2, 3]].long().T

        t_cls = t[:, 1].long()
        t_xy = t[:, [2, 3]] - t[:, [2, 3]].long().float()
        
        if self.wh_mode == 'v3':
            t_wh = torch.log(t[:, [4, 5]] / anchors[an_idx])
        elif self.wh_mode == 'v5':
            t_wh = torch.log2( torch.sqrt( t[:, [4, 5]] / anchors[an_idx] ) ) / 2. + 0.5
            # t_wh = torch.sqrt(t[:, [4, 5]] / anchors[an_idx]) / 2.

        return t, (im_idx, an_idx, j_idx, i_idx), t_cls, t_xy, t_wh