import numpy as np 
import torch
import math
import time

import torch


class PriorBox(object):
    def __init__(self, ):
        
        self.img_dim = 320
        self.strides = [8, 16, 32]
        self.grids = [int(self.img_dim / s) for s in self.strides]

        # anchor size
        self.base_sizes = [60, 120, 180]
        self.area_rarios = [[2, ], [2, ], [2, ]]
        self.aspect_rarios = [[2, ], [2, 3], [2, 3]]

    def __call__(self, ):
        '''
        generate prior bbox, [cx, cy, w, h] and normalized
        return: torch.Tensor
        '''
        anchors = []
        for i, grid in enumerate(self.grids):
            
            # center_x center_y
            yy, xx = np.meshgrid(range(grid), range(grid))
            # yy, xx = torch.meshgrid((torch.arange(grid), torch.arange(grid)))
            cx = (xx + 0.5) / grid
            cy = (yy + 0.5) / grid
        
            # width height
            whs = []
            base = self.base_sizes[i] / self.img_dim
            whs += [[base, base]]

            for ar in self.aspect_rarios[i]:
                whs += [[base * math.sqrt(ar), base / math.sqrt(ar)]]
                whs += [[base / math.sqrt(ar), base * math.sqrt(ar)]]
            
            for ar in self.area_rarios[i]:
                whs += [[base * math.sqrt(ar), base * math.sqrt(ar)]]
                whs += [[base / math.sqrt(ar), base / math.sqrt(ar)]]

            whs = np.array(whs)
            # whs = torch.from_numpy(np.array(whs))
            # print(len(whs))

            # n = xx.shape[0] * xx.shape[1]
            priors = np.zeros((np.prod(cx.shape), len(whs), 4))
            # print(priors.shape)
            # priors = torch.zeros((n, len(whs), 4))
            priors[:, :, 0] = cx.reshape(-1, 1)
            priors[:, :, 1] = cy.reshape(-1, 1)
            priors[:, :, 2:] = whs

            anchors += [priors.reshape(-1, 4)]
            # print(np.prod(cx.shape) * len(whs))

        anchors = np.concatenate(anchors, axis=0)
        # anchors = torch.cat(anchors, dim=0)
        # print(anchors.shape)
        anchors = torch.from_numpy(anchors).to(dtype=torch.float)
        # anchors.clamp_(max=1., min=0.)
        # print('anchors: ', anchors.shape)
        return anchors