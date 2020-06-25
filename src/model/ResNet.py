"""
ResNet module of MetaChrom
"""
__author__ = "Ben Lai"
__copyright__ = "Copyright 2020, TTIC"
__license__ = "GNU GPLv3"


import torch
import torch.nn as nn
from torch.autograd import Variable

def conv1d(in_channels, out_channels, stride = 1, kernel_size = 7):
    return nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size,
            padding = 3, bias = True, stride = stride)


class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv1d(in_channels, out_channels, stride = stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace = False)
        self.conv2 = conv1d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_target = 919):
        super(ResNet, self).__init__()
        self.in_channels = 4
        self.conv_kernel = 8
        self.pool_kernel = 4
        self.inplanes = 48

        self.conv1 = conv1d(self.in_channels,self.inplanes, stride = 1, kernel_size= 7)
        self.bn1 = nn.BatchNorm1d(48)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 3, padding = 0)

        self.layer1 = self._make_layer(block,96 , layers[0])
        self.layer2 = self._make_layer(block,128, layers[1], stride=1)
        self.layer3 = self._make_layer(block,256, layers[2], stride=1)
        self.layer4 = self._make_layer(block,512, layers[3], stride=1)

        self.conv_out = 512
        self.channel_out = 4

        self.classifier = nn.Sequential(
                nn.Linear(self.conv_out * self.channel_out, 1000),
                nn.ReLU(inplace = True),
                nn.Linear(1000, num_target),
                nn.ReLU(inplace = True),
                nn.Linear(num_target, num_target),
                nn.Sigmoid()
                )

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size = 1,stride = stride,
                        bias = False),
                    nn.BatchNorm1d(planes * block.expansion),
                    )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), self.conv_out * self.channel_out)
        x = self.classifier(x)
        return x

def ResMod(num_target):
    model = ResNet(ResidualBlock, [2,2,2,2], num_target = num_target)
    return model

