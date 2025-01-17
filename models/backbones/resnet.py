# models/backbones/resnet.py

import torch
import torch.nn as nn
import torchvision

class ResNet(nn.Module):
    def __init__(self,
                 model_name='resnet18',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[]):
        super().__init__()

        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None

        # e.g. load a resnet50 or resnet18 from torchvision
        if 'resnet18' in model_name.lower():
            self.model = torchvision.models.resnet18(weights=weights)
            out_channels = 512
        elif 'resnet50' in model_name.lower():
            self.model = torchvision.models.resnet50(weights=weights)
            out_channels = 2048
       

        
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
            out_channels //= 2  #
        if 3 in layers_to_crop:
            self.model.layer3 = None
            out_channels //= 2


        if layers_to_freeze >= 0:
            self.model.conv1.requires_grad_(False)
            self.model.bn1.requires_grad_(False)
        if layers_to_freeze >= 1:
            self.model.layer1.requires_grad_(False)
        if layers_to_freeze >= 2:
            self.model.layer2.requires_grad_(False)

        self.out_channels = out_channels

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)

        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)

        return x