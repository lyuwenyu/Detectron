
import _init_path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dets.data.dataset import DatasetYolov5
from dets.detector import DETRDetector


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dataset = DatasetYolov5('../../dataset/coco128/images/train2017/')



detr = DETRDetector()
print(detr)


data = torch.rand(8, 3, 1024, 640)

outputs = detr(data)

for out in outputs:
    print(out.shape)
