import _init_path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import dets 
from dets.detector import YOLOV3Detector
from dets.data.dataset import DatasetYolov5

from dets.utils.postprocess import non_max_suppression

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt



dataset = DatasetYolov5('../../dataset/coco128/images/train2017/')

_im = dataset[0][0]

im = torch.tensor(_im, dtype=torch.float32)


yolov3 = YOLOV3Detector()
yolov3.load_state_dict(torch.load('./yolov3.pt'))
yolov3.eval()

out = yolov3(im)
print(out.shape)
print(out.sum())

out = non_max_suppression(out, conf_thres=1e-25, iou_thres=0.01, multi_label=False)

# plt.imshow(_im[0].transpose(1, 2, 0) * dataset.std + dataset.mean)
# plt.show()

out = out[0].cpu().detach().numpy()
out = out[:, :4]
# out[:, [0, 1]] = out[:, [0, 1]] - out[:, [2, 3]] / 2
# out[:, [2, 3]] = out[:, [0, 1]] + out[:, [2, 3]]
np.clip(out, 0, 640-1, out=out)


im = Image.fromarray( ((_im[0].transpose(1, 2, 0) * dataset.std + dataset.mean) * 255).astype(np.uint8) )
draw = ImageDraw.Draw(im)
for b in out:
    draw.rectangle(b, outline='red')
    
plt.imshow(im)
plt.show()
