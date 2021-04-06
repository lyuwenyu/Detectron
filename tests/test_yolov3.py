import _init_path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import dets 
from dets.detector import YOLOV3Detector
from dets.data.dataset import DatasetYolov5


bbox = [0, 45, 0.479492, 0.688771, 0.955609, 0.5955,
        0, 45, 0.736516, 0.247188, 0.498875, 0.476417,
        0, 50, 0.637063, 0.732938, 0.494125, 0.510583,
        0, 45, 0.339438, 0.418896, 0.678875, 0.7815,
        0, 49, 0.646836, 0.132552, 0.118047, 0.096937,
        0, 49, 0.773148, 0.129802, 0.090734, 0.097229,
        0, 49, 0.668297, 0.226906, 0.131281, 0.146896,
        0, 49, 0.642859, 0.079219, 0.148063, 0.148062
    ]
targets = torch.tensor(bbox).view(-1, 6)

data = torch.rand(1, 3, 640, 640)
targets[:, 3:] = targets[:, 3:]



# output = yolov3(data, targets)
# try:
#     print(output.shape, output.sum())
# except:
#     print(output['loss'].item())

# print(yolov3)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

dataset = DatasetYolov5('/Users/lvwenyu01/Desktop/workspace/dataset/coco128/images/train2017/')
dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)

yolov3 = YOLOV3Detector().to(device=device)
yolov3.train()

# optimizer = optim.Adam(yolov3.parameters(), lr=0.001)

yolov3.model[0].eval()
for p in yolov3.model[0].parameters():
    p.requires_grad = False

optimizer = optim.SGD(yolov3.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.5)



for _ in range(3):

    for data, label in dataloader:
        data = data.to(device=device)
        label = label.to(device=device)

        losses = yolov3(data, label)
        
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()
        
        print(losses['loss'].item(), losses['lbox'], losses['lobj'], losses['lcls'], )


yolov3 = yolov3.to(torch.device('cpu'))
torch.save(yolov3.state_dict(), './yolov3.pt')

