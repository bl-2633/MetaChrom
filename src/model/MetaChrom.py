
"""
MetaChrom model
"""
__author__ = "Ben Lai"
__copyright__ = "Copyright 2020, TTIC"
__license__ = "GNU GPLv3"

import torch
import torch.nn as nn
import numpy as np
from .ResNet import ResMod
from .MetaFeat import MetaFeat

class MetaChrom(torch.nn.Module):
    def __init__(self, num_target, load_base = False, base_path = None):
        super(MetaChrom, self).__init__()
        self.base_path = base_path
        self.classifier = nn.Sequential(
                nn.Linear( 919 + num_target, num_target),
                nn.ReLU(inplace=True),
                nn.Linear(num_target, num_target),
                nn.Sigmoid())

        self.base_model = MetaFeat(919)
        if load_base:
            device = torch.device('cpu')
            check_point = torch.load(self.base_path, map_location =device)
            self.base_model.load_state_dict(check_point['state_dict'])
        self.seq_model = ResMod(num_target=num_target)

    def forward(self,x):
        feat= self.base_model(x)
        out = self.seq_model(x)
        feat_cat =  torch.cat((feat, out), 1)
        out = self.classifier(feat_cat)
        return out