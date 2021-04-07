import init_path

import argparse

import torch
import numpy as np 


parser = argparse.ArgumentParser()
parser.add_argument('--epoches', type=int, default=100, )
parser.add_argument('--batch-size', type=int, default=16, )
parser.add_argument('--lr', type=float, default=0.01, )
args = parser.parse_args()
print(args)



def train(epoch, model, dataloader, optimizer):
    model.train()

    losses = []

    for i, (data, target) in enumerate(dataloader):

        loss = model(data, target)

        optimizer.zeor_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if i % 10 == 0:
            print(epoch, i, np.mean(losses[i-9:i+1]))
            # losses.clear()




def test(model, dataloader, ):
    model.eval()

    pass



if __name__ == '__main__':
    

    pass
