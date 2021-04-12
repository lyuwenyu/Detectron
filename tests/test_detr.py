
import _init_path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dets.data.dataset import DatasetYolov5
from dets.detector import DETRDetector


# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# detr = DETRDetector().to(device)
# print(detr)

# data = torch.rand(2, 3, 1024, 640).to(device)

# outputs = detr(data)

# for k in outputs:
#     print(outputs[k].shape)


# --------

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
dataset = DatasetYolov5('../../dataset/coco128/images/train2017/')
dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn, shuffle=True)

detr = DETRDetector().to(device)
detr.train()


optimizer = optim.SGD(detr.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[18, 30], gamma=0.2)

    

for _ in range(20):

    for data, label in dataloader:
        data = data.to(device=device)
        label = label.to(device=device)
        
        losses = detr(data, label)
        
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()
        
        print(losses['loss'].item(), losses['lbox'],  losses['lobj'], losses['lcls'], )


ssd = ssd.to(torch.device('cpu'))
torch.save(ssd.state_dict(), './detr.pt')

