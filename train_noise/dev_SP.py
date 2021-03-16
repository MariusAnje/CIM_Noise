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
from models import LeNet_Gaussian as Net


net = Net()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

BS = 128

trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True,
                                        download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                        shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                    download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                        shuffle=False, num_workers=4)


   
model = Net()
keys = [0, 10, 23, 51, 81, 118, 209]
keys = [0, 11, 54, 80, 117, 240]
keys = [9900, 9901, 9904, 9905, 9909]
file_key = 240
# state_dict = torch.load(f"GCONV_state_dict_{file_key}.pt", map_location=device)
# state_dict = torch.load(f"step_st/GCONV_state_dict_tmp_best_{file_key}.pt", map_location=device)
state_dict = torch.load(f"noise_trace/300-1/GCONV_state_dict_noise_1615317550.451319_tmp_best.pt{file_key}", map_location=device)

oModel = Net()
oModel.load_state_dict(state_dict)
oModel.to(device)
oModel.clear_noise()
correct = 0
total = 0
GT = torch.zeros(len(testset))
for i, data in enumerate(testloader):
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
    GT[i*BS:(i+1)*BS] = predicted
print(correct/total)


noise = (0, 0.1) # (0.1 -> 48%) (0.07 -> 78%) (0.04 -> 95%)
# noise = (0,0)
acc_list = []
nRuns = 10000
robust_table = torch.zeros(total).to(device)
table2D = torch.zeros(nRuns, total).to(torch.int8)
pModel = Net()
pModel.load_state_dict(state_dict)
pModel.to(device)
# for run_index in tqdm.tqdm(range(nRuns), leave=False):
for run_index in range(nRuns):
    pModel.set_noise(noise[0], noise[1])
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(testloader):
            images, labels = data
            labels = GT[i*BS:(i+1)*BS]
            images, labels = images.to(device), labels.to(device)
            outputs = pModel(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_table = predicted == labels
            robust_table[i*BS:(i+1)*BS] += correct_table
            table2D[run_index, i*BS:(i+1)*BS] = correct_table
            correct += correct_table.sum().item()
    acc_list.append(correct/total)
print(np.mean(acc_list))
print(robust_table.mean()/100)
print(robust_table[:20])
torch.save(robust_table.to(torch.device("cpu")), f"gaussian_{noise[1]}_table_{file_key}.pt")
torch.save(table2D.to(torch.device("cpu")), f"gaussian_{noise[1]}_2D_{file_key}.pt")
