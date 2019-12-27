import torch
from torch import nn

class SameConv(nn.Module):
    def __init__(self, num_layers, kernel_size, in_channels, out_channels):
        super(SameConv, self).__init__()
        layers = []
        for i in range(num_layers):
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2), bias=False)
            layers += [conv2d]
            in_channels = out_channels

        self.feature = nn.Sequential(*layers)

    def forward(self, x):
        return self.feature(x)

class SameConv_ReLU(nn.Module):
    def __init__(self, num_layers, kernel_size, in_channels, out_channels):
        super(SameConv_ReLU, self).__init__()
        layers = []
        for i in range(num_layers):
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2), bias=False)
            if i != num_layers - 1:
                layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d]
            in_channels = out_channels

        self.feature = nn.Sequential(*layers)

    def forward(self, x):
        return self.feature(x)

if __name__ == "__main__":
    num_layers = 3
    kernel_size = 5
    in_channels = 3
    out_channels = 3
    model = SameConv(num_layers, kernel_size, in_channels, out_channels)
    print(model)