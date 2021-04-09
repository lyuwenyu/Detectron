import _init_path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import dets 
from dets.detector import SSDDetector
from dets.data.dataset import DatasetYolov5

from dets.utils.postprocess import non_max_suppression

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt



dataset = DatasetYolov5('../../dataset/coco128/images/train2017/')

_im = dataset[0][0]

im = torch.tensor(_im, dtype=torch.float32)


ssd = SSDDetector()
ssd.load_state_dict(torch.load('./ssd.pt'))
ssd.eval()

out = ssd(im)
print(out.sum())
out = out[0].cpu().detach().numpy()
print(out.shape)

# out[:, :, 4:5] = 1 - out[:, :, 4:5]
# out = non_max_suppression(out, conf_thres=0.4, iou_thres=0.2, multi_label=False)

# plt.imshow(_im[0].transpose(1, 2, 0) * dataset.std + dataset.mean)
# plt.show()

# print(out[:, 4:])
print((out[:, 4:].argmax(axis=-1) > 0).sum())

print(out[:, 4:].max(axis=-1).shape)

out = out[(out[:, 4:].argmax(axis=-1) > 0) & (out[:, 4:].max(axis=-1) > 0.8)]

# out = out[0].cpu().detach().numpy()
out = out[:, :4] * 640
out[:, [0, 1]] = out[:, [0, 1]] - out[:, [2, 3]] / 2
out[:, [2, 3]] = out[:, [0, 1]] + out[:, [2, 3]]
np.clip(out, 0, 640-1, out=out) 


im = Image.fromarray( ((_im[0].transpose(1, 2, 0) * dataset.std + dataset.mean) * 255).astype(np.uint8) )
draw = ImageDraw.Draw(im)
for b in out:
    draw.rectangle(b, outline='red')
    
plt.imshow(im)
plt.show()
