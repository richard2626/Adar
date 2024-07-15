import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import argparse
# from torchsummary import summary

# Homework 1: 
# calculate the input/output size of each layer 
# calculate the total number of patameters

# In lec2_hw1.py, calculate the input/output size of each
# layer in NetHW and the total # of parameters of this
# model.
 
class NetHW(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 8,  3, 1, 1, bias=False)
        self.conv_2 = nn.Conv2d(8, 8,  5, 1, 2, bias=False)
        self.conv_3 = nn.Conv2d(8, 16, 3, 1, 1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(4*4*16, 10, bias=False)

    def forward(self, x):           # x : [B, 3, 32, 32]
        out = self.conv_1(x)        # [B, 3, 32, 32]  -> [B, 8, 32, 32]
        out = self.maxpool(out)     # [B, 8, 32, 32]  -> [B, 8, 16, 16]
        out = self.conv_2(out)      # [B, 8, 16, 16]  -> [B, 8, 16, 16]
        out = self.maxpool(out)     # [B, 8, 16, 16]  -> [B, 8, 8, 8]
        out = self.conv_3(out)      # [B, 8, 8, 8]  -> [B, 16, 8, 8]
        out = self.maxpool(out)     # [B, 16, 8, 8]  -> [B, 16, 4, 4]
        out = self.flatten(out)     # [B, 16, 4, 4]  -> [B, 4*4*16]
        out = self.fc_1(out)        # [B, 4*4*16]  -> [B, 10]
        return out

# Total number of patameters: 
# conv_1: 3*8*3*3 = 216
# conv_2: 8*8*5*5 = 1600
# conv_3: 8*16*3*3 = 1152
# fc_1: 4*4*16*10 = 2560
# Total: 5528





