import _init_path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dets.data.dataset import DatasetYolov5
from dets.detector import DETRDetector

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt



device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
dataset = DatasetYolov5('../../dataset/coco128/images/train2017/')


_im = dataset[0][0]
im = torch.tensor(_im, dtype=torch.float32)


detr = DETRDetector().to(device)
detr.load_state_dict(torch.load('./detr.pt'))
detr.eval()


out = detr(im)
bbox = out['pred_boxes'].cpu().data.numpy()[0] * 640
print(bbox.shape)


# print(bbox)

im = Image.fromarray( ((_im[0].transpose(1, 2, 0) * dataset.std + dataset.mean) * 255).astype(np.uint8) )
draw = ImageDraw.Draw(im)
for b in bbox:
    _b =( b[0] - b[2]/2, b[1] - b[3]/2, b[0] + b[2]/2, b[1] + b[3]/2, )
    draw.rectangle(_b, outline='red')
    
plt.imshow(im)
plt.show()