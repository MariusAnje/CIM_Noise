import torch
from torch import nn
import torch.nn.functional as F
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
import pickle
import tqdm
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
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


net = Net()


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

BS = 128

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                        shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                        shuffle=False, num_workers=4)


   
model = Net()

state_dict = torch.load("GCONV_state_dict.pt", map_location=device)

oModel = Net()
oModel.load_state_dict(state_dict)
oModel.to(device)
correct = 0
total = 0
for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    # images = images.view(-1,784)
    # mean = torch.zeros(1,2)
    # std = torch.ones(1,2)
    # images = torch.normal(mean,std)
    outputs = oModel(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print(correct/total)

for file_index in range(1):
    # noise = (0,4e-2)
    noise = (0,0)
    pOutput = []
    nRuns = 1
    import tqdm
    for _ in tqdm.tqdm(range(nRuns), leave=False):
    # for _ in tqdm.tqdm_notebook(range(nRuns), leave=False):
        pModel = Net()
        pModel.load_state_dict(state_dict)
        pState = pModel.state_dict()
        for noise_index, key in enumerate(state_dict.keys()):
            if key.find("weight") != -1:
                size = state_dict[key].size()
                mean, std = noise
                sampled_noise = torch.randn(size) * std + mean
                pState[key] = pState[key].data + sampled_noise
        pModel.load_state_dict(pState)
        pModel.to(device)

        pOutputItem = []
        with torch.no_grad():
            # for data in tqdm.tqdm(testloader, leave=False):
            for data in testloader:
                
                images, _ = data
                outputs = pModel(images)
                _, predicted = torch.max(outputs.data, 1)
                pOutputItem.append(outputs.detach().cpu().numpy())
        tmp = np.zeros([10000, 10])
        for i in range(10000//BS + 1):
            tmp[i*BS:i*BS+len(pOutputItem[i]),:] = pOutputItem[i]
        pOutput.append(tmp)

    np.savez_compressed(f"output0_{file_index:02d}", pOutput)