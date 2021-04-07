


import _init_path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dets.utils import matcher


# x y w h
bbox = [0, 45, 0.479492, 0.688771, 0.955609, 0.5955,
        0, 45, 0.736516, 0.247188, 0.498875, 0.476417,
        0, 50, 0.637063, 0.732938, 0.494125, 0.510583,
        0, 45, 0.339438, 0.418896, 0.678875, 0.7815,
        0, 49, 0.646836, 0.132552, 0.118047, 0.096937,
        0, 49, 0.773148, 0.129802, 0.090734, 0.097229,
        0, 49, 0.668297, 0.226906, 0.131281, 0.146896,
        0, 49, 0.642859, 0.079219, 0.148063, 0.148062]

bbox = torch.tensor(bbox).view(-1, 6)
data = torch.rand(1, 3, 640, 640)


target = {}
target['labels'] = bbox[:, 1].long()
target['boxes'] = bbox[:, 2:]

outputs = {}
outputs['pred_logits'] = torch.rand(1, 100, 80)
outputs['pred_boxes'] = torch.rand(1, 100, 4)

m = matcher.HungarianMatcher()
index = m(outputs, [target])

print(index)
