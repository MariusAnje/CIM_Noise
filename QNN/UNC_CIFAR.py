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
import resCIFAR

device = torch.device("cuda:0")
BS = 128
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
#     normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

transform = transforms.Compose(
[transforms.ToTensor(),
    normalize])
# trainset = torchvision.datasets.CIFAR10(root='~/Private/data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='~/Private/data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=4)

# if args.model == "NS":
#     model = modelNeuroSIM.NeuroSIM_BN_Model()
# elif args.model == "RES":
#     model = resnet.resnet56_cifar(num_classes=10)

# model = modelNeuroSIM.NeuroSIM_BN_Model()

# state_dict = torch.load(f"CIFAR10_BN_Aug_{args.model}.pt", map_location=device)

Net = resCIFAR.resnet110
state_dict = torch.load(f"resnet110-1d1ed7c2.th", map_location=device)
state_dict = state_dict["state_dict"]
state_dict = {k[7:]: v for k, v in state_dict.items()}


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
        # GT = []
        with torch.no_grad():
            # for data in tqdm.tqdm(testloader, leave=False):
            for data in testloader:
                
                images, labels = data
                images = images.to(device)
                outputs = pModel(images)
                _, predicted = torch.max(outputs.data, 1)
                pOutputItem.append(outputs.detach().cpu().numpy())
                # GT += labels.numpy().tolist()
        tmp = np.zeros([10000, 10])
        for i in range(10000//BS + 1):
            tmp[i*BS:i*BS+len(pOutputItem[i]),:] = pOutputItem[i]
        pOutput.append(tmp)

    np.savez_compressed(f"Res110_0_{file_index:02d}", pOutput)
    # np.save("CIFAR10GT", GT)
