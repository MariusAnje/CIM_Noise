import torch
from torch import nn
import torch.nn.functional as F
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
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

BS = 256

trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True,
                                        download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                        shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                    download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                        shuffle=False, num_workers=4)



model = Net()
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0)

acc_name_list = [9900, 9901, 9904, 9905, 9909]
acc_name = 9360

# state_dict = torch.load(f"GCONV_state_dict_{acc_name}.pt", map_location=device)
step_names = [0, 4, 8, 15]
step_name = 15
state_dict = torch.load(f"step_st/GCONV_state_dict_tmp_best_{step_name}.pt", map_location=device)

model.load_state_dict(state_dict)
model.eval()

correct = 0
total = 0
adv_grad = torch.zeros(0,1,28,28)
consis_table = torch.zeros(0,dtype=torch.long)

for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    # print(images.size())
    images.requires_grad_()
    optimizer.zero_grad()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    consis_table = torch.cat([consis_table, predicted])
    loss = criteria(outputs, predicted)
    loss.backward()
    adv_grad = torch.cat([adv_grad, images.grad])
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print(correct/total)

s_adv_grad = adv_grad.sign()
ptb_amp_choices = np.arange(0.001,0.5,0.001)
# ptb_amp_choices = [0.1]
robust_dict = {}

for ptb_amp in ptb_amp_choices:
    robust_table = torch.zeros(total, dtype=torch.int8)
    oModel = Net()
    oModel.load_state_dict(state_dict)
    oModel.to(device)
    oModel.eval()
    correct = 0
    total = 0
    for i, data in enumerate(testloader):
        images, labels = data
        noise = s_adv_grad[i*BS:(i+1)*BS] * ptb_amp
        cons = consis_table[i*BS:(i+1)*BS]
        images += noise
        images, labels = images.to(device), labels.to(device)
        outputs = oModel(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_table = predicted == cons
        # print(correct_table)
        robust_table[i*BS:(i+1)*BS] += correct_table
        correct += correct_table.sum().item()
    print(f"amp: {ptb_amp:.3f}, acc:{correct/total}")
    # print(robust_table[:30].numpy().tolist())
    robust_dict[ptb_amp] = robust_table
# torch.save(robust_dict, f"consist_dict_{acc_name}.pt")
torch.save(robust_dict, f"consist_dict_{step_name}.pt")