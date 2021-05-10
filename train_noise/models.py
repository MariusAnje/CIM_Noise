import torch
from torch import nn
import torch.nn.functional as F
import torch
from modules import GConv2d, GLinear, QuantGConv2d,  QuantGLinear

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 7 * 7, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class LeNet_Gaussian(nn.Module):
    def __init__(self):
        super(LeNet_Gaussian, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = GConv2d(in_channels = 1, out_channels = 6, kernel_size = 3, padding=1)
        self.conv2 = GConv2d(6, 16, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = GLinear(16 * 7 * 7, 120)  # 6*6 from image dimension
        self.fc2 = GLinear(120, 84)
        self.fc3 = GLinear(84, 10) 
    
    def set_noise(self, mean, std):
        for m in self.modules():
            if isinstance(m, GConv2d) or isinstance(m, GLinear):
                m.set_noise(mean, std)

    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, GConv2d) or isinstance(m, GLinear):
                m.clear_noise()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class LeNet_Quant_Gaussian(nn.Module):
    def __init__(self):
        super(LeNet_Quant_Gaussian, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        nbits = 8
        ndevice = 1
        var = 0.40
        self.conv1 = QuantGConv2d(nbits, ndevice, var, in_channels = 1, out_channels = 6, kernel_size = 3, padding=1)
        self.conv2 = QuantGConv2d(nbits, ndevice, var, 6, 16, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = QuantGLinear(nbits, ndevice, var, 16 * 7 * 7, 120)  # 6*6 from image dimension
        self.fc2 = QuantGLinear(nbits, ndevice, var, 120, 84)
        self.fc3 = QuantGLinear(nbits, ndevice, var, 84, 10) 
    
    def set_noise(self, mean, std):
        for m in self.modules():
            if isinstance(m, QuantGConv2d) or isinstance(m, QuantGLinear):
                m.set_noise()

    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, QuantGConv2d) or isinstance(m, QuantGLinear):
                m.clear_noise()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NSIM_Quant_Gaussian(nn.Module):
    def __init__(self):
        super(NSIM_Quant_Gaussian, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        nbits = 8
        ndevice = 1
        var = 0.35
        # var = 0.0

        self.conv1 = QuantGConv2d(nbits, ndevice, var,   3, 128, 3, padding=1)
        self.conv2 = QuantGConv2d(nbits, ndevice, var, 128, 128, 3, padding=1)
        self.conv3 = QuantGConv2d(nbits, ndevice, var, 128, 256, 3, padding=1)
        self.conv4 = QuantGConv2d(nbits, ndevice, var, 256, 256, 3, padding=1)
        self.conv5 = QuantGConv2d(nbits, ndevice, var, 256, 512, 3, padding=1)
        self.conv6 = QuantGConv2d(nbits, ndevice, var, 512, 512, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        # an affine operation: y = Wx + b
        self.fc1 = QuantGLinear(nbits, ndevice, var, 8192, 1024)  # 6*6 from image dimension
        self.fc2 = QuantGLinear(nbits, ndevice, var, 1024, 10)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
    
    def set_noise(self, mean, std):
        for m in self.modules():
            if isinstance(m, QuantGConv2d) or isinstance(m, QuantGLinear):
                m.set_noise()

    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, QuantGConv2d) or isinstance(m, QuantGLinear):
                m.clear_noise()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features