
import torch
import torch.nn as nn
import torchvision.models as models



def resnet50(pretrained=True):
    '''reset50
    '''
    net = models.resnet50(pretrained=pretrained)    
    net = nn.Sequential(*list(net.children())[:-2]) 
    return net


def resnet101(pretrained=True):
    '''reset101
    '''
    net = models.resnet50(pretrained=pretrained)    
    net = nn.Sequential(*list(net.children())[:-2]) 
    return net



def resnet50_16(pretrained=True):
    '''reset50
    '''
    net = models.resnet50(pretrained=pretrained)    
    net = list(net.children())[:-2]

    layer4_0 = net[-1][0]
    layer4_0.conv2 = nn.Conv2d(512, 512, 3, stride=1, padding=2, dilation=2, bias=False)
    layer4_0.downsample[0] = nn.Conv2d(1024, 2048, 1, stride=1, bias=False)
    net[-1][0] = layer4_0

    return nn.Sequential(*net)



def darknet(pretrained=False):
    pass



class Resnet50(nn.Module):
    def __init__(self, pretrained=False, num_layers=3):
        super().__init__()
        net = models.resnet50(pretrained=pretrained)    
        net = nn.Sequential(*list(net.children())[:-2]) 

        out_channels_list = [512, 1024, 2048]

        self.net = net

        assert isinstance(num_layers, int) and num_layers <= 3, ''
        self.num_layers = num_layers
        self.out_channels_list = out_channels_list[-num_layers:]

    def forward(self, data):
        
        outputs = []
        for m in self.net.children():
            data = m(data)
            outputs.append(data)

        return outputs[-self.num_layers:]



if __name__ == '__main__':


    net = Resnet50()

    for _ in range(10):

        data = 2 * torch.rand(1, 3, 640, 640) - 1
        outs = net(data)    
        for o in outs:
            print(o.shape)
            print((o < 0).sum())
