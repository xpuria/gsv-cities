# models/aggregators/avgpool.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgPool(nn.Module):
    """
    A simple aggregator:
      1) Global average pool -> [N, C, 1, 1]
      2) Flatten -> [N, C]
      3) L2 normalize -> [N, C] (unit vectors)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1)  
        x = x.view(x.size(0), -1)       
        x = F.normalize(x, p=2, dim=1)
        return x