import torch
from torch import nn
from quantNN import *

class NeuroSIM_Model(nn.Module):
    def __init__(self):
        super(NeuroSIM_Model, self).__init__()
        N = 8
        m = 5
        self.features = nn.Sequential(
            QConv2d(N, 5, 3, 128, 3, padding=1),
            nn.ReLU(),
            QConv2d(N, 5, 128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            QConv2d(N, 5, 128, 256, 3, padding=1),
            nn.ReLU(),
            QConv2d(N, 4, 256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            QConv2d(N, 4, 256, 512, 3, padding=1),
            nn.ReLU(),
            QConv2d(N, 4, 512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            QLinear(N, 5, 8192, 1024),
            nn.ReLU(),
            QLinear(N, 5, 1024, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(-1,8192))

class NeuroSIM_BN_Model(nn.Module):
    def __init__(self):
        super(NeuroSIM_BN_Model, self).__init__()
        N = 8
        m = 5
        self.features = nn.Sequential(
            QConv2d(N, 5, 3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            QConv2d(N, 5, 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            QConv2d(N, 5, 128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            QConv2d(N, 5, 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            QConv2d(N, 5, 256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QConv2d(N, 5, 512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            QLinear(N, 5, 8192, 1024),
            nn.ReLU(),
            QLinear(N, 5, 1024, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(-1,8192))

class Visual_Model(nn.Module):
    def __init__(self):
        super(Visual_Model, self).__init__()
        N = 16
        m = 5
        self.features = nn.Sequential(
            QConv2d(N, 5, 3, 128, 3, padding=1, is_print = True),
            nn.ReLU(),
            QConv2d(N, 5, 128, 128, 3, padding=1, is_print = True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            QConv2d(N, 5, 128, 256, 3, padding=1, is_print = True),
            nn.ReLU(),
            QConv2d(N, 4, 256, 256, 3, padding=1, is_print = True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            QConv2d(N, 4, 256, 512, 3, padding=1, is_print = True),
            nn.ReLU(),
            QConv2d(N, 4, 512, 512, 3, padding=1, is_print = True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            QLinear(N, 4, 8192, 1024, is_print = True),
            nn.ReLU(),
            QLinear(N, 4, 1024, 10, is_print = True)
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(-1,8192))

if __name__ == "__main__":
    model = NeuroSIM_Model()
    a = torch.Tensor(1,3,32,32)
    print(a.size())
    print(model(a))
