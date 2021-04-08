import numpy as np 
import torch
import math
import time

import torch


class PriorBox(object):
    def __init__(self, img_size=640, strides=8, ):
        '''
        img_size [w, h]
        stride [8, 16, 32]
        returns:
            [(w/8 * h/8) * a * 4, (w/16 * h/16) * a * 4]
        '''
        if not isinstance(img_size, (list, tuple)):
            img_size = (img_size, img_size)
        
        if not isinstance(strides, (list, tuple)):
            strides = (strides, )

        self.img_size = img_size
        self.strides = strides

        # anchor size
        self.base_sizes = [(60, 60), (160, 160), (320, 320)]
        self.aspect_rarios = [(2, ), (2, ), (2, )]
        self.area_rarios = [(2, ), (2, ), (2, )]
        self.num_anchors = [5, 5, 5]

    def __call__(self, ):
        '''
        generate prior bbox, [cx, cy, w, h] and normalized, respect to orgin image
        return: torch.Tensor
        '''
        anchors = []

        for i, s in enumerate(self.strides):
            
            grid_w = int(self.img_size[0] / s)
            grid_h = int(self.img_size[1] / s)

            # np.meshgrid is different from torch.meshgrid
            gi, gj = np.meshgrid(range(grid_w), range(grid_h))

            cx = (gi + 0.5) / grid_w
            cy = (gj + 0.5) / grid_h

            whs = []

            base_w = self.base_sizes[i][0] / self.img_size[0]
            base_h = self.base_sizes[i][1] / self.img_size[1]

            whs.append([base_w, base_h])

            for ar in self.aspect_rarios[i]:
                whs.append( [base_w * math.sqrt(ar), base_h / math.sqrt(ar)] )
                whs.append( [base_w / math.sqrt(ar), base_h * math.sqrt(ar)] )
            
            for ar in self.area_rarios[i]:
                whs += [[base_w * math.sqrt(ar), base_h * math.sqrt(ar)]]
                whs += [[base_w / math.sqrt(ar), base_h / math.sqrt(ar)]]

            whs = np.array(whs)

            priors = np.zeros((np.prod(cx.shape), len(whs), 4))

            priors[:, :, 0] = cx.reshape(-1, 1)
            priors[:, :, 1] = cy.reshape(-1, 1)
            priors[:, :, 2:] = whs.reshape(-1, 2)

            anchors += [priors.reshape(-1, 4)]

        anchors = np.concatenate(anchors, axis=0)

        return anchors



if __name__ == '__main__':


    bbox = PriorBox(img_size=(640, 512), strides=(8, 16))()

    print(bbox.shape)
    print(bbox[-10:])
