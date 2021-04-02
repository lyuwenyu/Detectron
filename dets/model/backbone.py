
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
    def __init__(self, pretrained=False):
        super().__init__()
        net = models.resnet50(pretrained=pretrained)    
        net = nn.Sequential(*list(net.children())[:-2]) 

        self.net = net
        self.out_channels_list = [512, 1024, 2048]

    def forward(self, data):
        
        outputs = []
        for m in self.net.children():
            data = m(data)
            outputs.append(data)

        return outputs[-3:]


if __name__ == '__main__':

    data = torch.rand(1, 3, 640, 640)
    net = resnet50_16(False)
    print(net(data).shape)


    net = Resnet50()
    outs = net(data)    
    for o in outs:
        print(o.shape)
