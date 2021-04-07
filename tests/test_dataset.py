import _init_path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import dets 
from dets.data.dataset import DatasetYolov5


dataset = DatasetYolov5('../../dataset/coco128/images/train2017/')
dataloader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate_fn)


dataset.show(0).show()

for data, label in dataloader:
   
    pass

