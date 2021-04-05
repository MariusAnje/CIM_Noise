import torch
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


class GConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(GConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        # self.noise = torch.zeros_like(self.conv.weight)
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.conv.weight)
    
    def set_noise(self, mean, std):
        self.noise = torch.randn_like(self.conv.weight)
    
    def forward(self, x):
        return F.conv2d(x, self.conv.weight + self.noise, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

class GLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GLinear, self).__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        # self.noise = torch.zeros_like(self.conv.weight)
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def set_noise(self, mean, std):
        self.noise = torch.randn_like(self.op.weight)
    
    def forward(self, x):
        return F.linear(x, self.op.weight, self.op.bias)



class AdvDataset():
    def __init__(self, total_set):
        self.total_set = total_set
        self.sens_table = np.ones(len(total_set))
    
    def sample(self, num, BS):
        T = np.where(self.sens_table)[0]
        if len(T) == 0:
            return [], T
        if num < len(T):
            T = np.random.choice(T, num, replace=False)
        subset = torch.utils.data.Subset(self.total_set, T)
        subloader = torch.utils.data.DataLoader(subset, batch_size=BS,
                                        shuffle=True, num_workers=2)
        return subloader, T
    
    def dual_draw(self, num, BS):
        S = np.where(self.sens_table)[0]
        R = np.where(self.sens_table == 0)[0]
        half = num // 2
        if half < len(S):
            S = np.random.choice(S, half, replace=False)
        if half < len(R):
            R = np.random.choice(R, half, replace=False)
        T = np.concatenate([R,S])
        subset = torch.utils.data.Subset(self.total_set, T)
        subloader = torch.utils.data.DataLoader(subset, batch_size=BS,
                                        shuffle=True, num_workers=2)
        return subloader, T
    
    def ballance_draw(self, num, BS):
        S = np.where(self.sens_table)[0]
        R = np.where(self.sens_table == 0)[0]
        half = min(num, len(S), len(R))

        if half != 0 and half < len(S):
            S = np.random.choice(S, half, replace=False)
        if half != 0 and half < len(R):
            R = np.random.choice(R, half, replace=False)
        T = np.concatenate([R,S])
        subset = torch.utils.data.Subset(self.total_set, T)
        subloader = torch.utils.data.DataLoader(subset, batch_size=BS,
                                        shuffle=True, num_workers=2)
        return subloader, T
    
    def whole_set(self, BS):
        T = list(range(len(self.total_set)))
        np.random.shuffle(T)
        subset = torch.utils.data.Subset(self.total_set, T)
        subloader = torch.utils.data.DataLoader(subset, batch_size=BS,
                                        shuffle=False, num_workers=2)
        return subloader, T