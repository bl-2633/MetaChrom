"""
Baseline CNN model for MetaChrom analysis
"""
__author__ = "Ben Lai"
__copyright__ = "Copyright 2020, TTIC"
__license__ = "GNU GPLv3"

import torch
import torch.nn as nn
import numpy as np

class CNN(torch.nn.Module):
    def __init__(self, num_target, sequence_length = 1000):
        super(CNN, self).__init__()
        self.conv_kernel = 8
        self.pool_kernel = 4

        self.H1 = 64
        self.H2 = 128
        self.H3 = 256
        self.input_channels = 4

        self.conv_net = nn.Sequential(
                nn.Conv1d(self.input_channels, self.H1, kernel_size = self.conv_kernel),
                nn.ReLU(inplace = True),
                nn.MaxPool1d(kernel_size = self.pool_kernel, stride=self.pool_kernel),

                nn.Conv1d(self.H1, self.H2, kernel_size = self.conv_kernel),
                nn.ReLU(inplace = True),
                nn.MaxPool1d(kernel_size = self.pool_kernel, stride=self.pool_kernel),


                nn.Conv1d(self.H2, self.H3, kernel_size = self.conv_kernel),
                nn.ReLU(inplace = True),
                nn.MaxPool1d(kernel_size = self.pool_kernel, stride=self.pool_kernel),
                )

        self.pool_kernel = float(self.pool_kernel)
        self.sequence_length = sequence_length
        self.n_channels = 13

        self.classifier = nn.Sequential(
                nn.Linear( 256 * self.n_channels, num_target),
                nn.Sigmoid()
                )

    def forward(self,x):
        out = self.conv_net(x)
        out = out.view(out.size(0), 256 * self.n_channels)
        out = self.classifier(out)
        
        return out

