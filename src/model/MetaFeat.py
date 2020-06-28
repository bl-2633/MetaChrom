"""
Meta-Feature extractor for MetaChrom
"""
__author__ = "Ben Lai"
__copyright__ = "Copyright 2020, TTIC"
__license__ = "GNU GPLv3"


import torch
import torch.nn as nn
import numpy as np

class MetaFeat(torch.nn.Module):
    def __init__(self, num_target):
        super(MetaFeat, self).__init__()
        self.conv_kernel = 8
        self.pool_kernel = 4

        self.H1 = 320
        self.H2 = 480
        self.H3 = 960
        self.input_channels = 4

        self.conv_net = nn.Sequential(
                nn.Conv1d(self.input_channels, self.H1, kernel_size = self.conv_kernel),
                nn.ReLU(inplace = True),
                nn.MaxPool1d(kernel_size = self.pool_kernel,stride=self.pool_kernel),
                nn.Dropout(p=0.2),

                nn.Conv1d(self.H1, self.H2, kernel_size = self.conv_kernel),
                nn.ReLU(inplace = True),
                nn.MaxPool1d(kernel_size = self.pool_kernel,stride=self.pool_kernel),
                nn.Dropout(p=0.2),

                nn.Conv1d(self.H2, self.H3, kernel_size = self.conv_kernel),
                nn.ReLU(inplace = True),
                nn.MaxPool1d(kernel_size = self.pool_kernel,stride=self.pool_kernel),
                nn.Dropout(p=0.5)
                )
        self.n_channels = 13
        self.classifier = nn.Sequential(
                nn.Linear(960 * self.n_channels, num_target),
                nn.ReLU(inplace=True),
                nn.Linear(num_target, num_target),
                nn.Sigmoid())

    def forward(self,x):
        out = self.conv_net(x)
        out = out.view(out.size(0), 960 * self.n_channels)
        out = self.classifier(out)
        return out

