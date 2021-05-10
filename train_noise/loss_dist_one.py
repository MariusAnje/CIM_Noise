import torch
from torch import nn
import torch.nn.functional as F
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
import numpy as np
from models import LeNet_Gaussian, LeNet_Quant_Gaussian
from modules import AdvDataset
from utils import *
import argparse
import time

def test(model, testloader, device, use_noise, mean=0, std=0):
    model.eval()
    correct = 0
    running_loss = 0.0
    total = 0
    with torch.no_grad():
        if use_noise:
            model.set_noise(mean,std)
        else:
            model.clear_noise()
        images, labels = the_input
        images, labels = images.to(device), labels.to(device)
        images = images.view(1,1,28,28)
        labels = labels.view(1)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        running_loss += criterion(outputs, labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        model.clear_noise()
        return correct/total, (running_loss/total).item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    parser.add_argument('--BS', action='store', type=int, default=128,
            help='input the batch size')
    parser.add_argument('--test_run', action='store', type=int, default=10, 
            help='input the number of runs for noisy test')
    parser.add_argument('--is_train', action='store', type=bool, default=False, 
            help='are we training?')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    BS = args.BS

    trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                            download=False, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='../data', train=False,
                                        download=False, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                            shuffle=False, num_workers=2)

    model = LeNet_Quant_Gaussian()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20 * len(trainloader), 0.1)
    adv_set = AdvDataset(trainset)
    mean, std = 0, 0

    model.to(device)
    model.clear_noise()
    print(args)
    start_time = time.time()

    if args.is_train:
        best_correct = 0
        for epoch in range(80):  # loop over the dataset multiple times
            model.train()
            running_loss = 0.0
            loader = tqdm(trainloader, leave=False)
            # loader = trainloader
            for _, data in enumerate(loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
            correct, test_loss = test(model, testloader, device, use_noise=False)
            print(f"acc: {correct:.4f}, loss: {test_loss:.4f}")
            if correct > best_correct:
                best_correct = correct
                torch.save(model.state_dict(), "GCONV_best.pt")

    state_dict = torch.load("GCONV_best.pt")
    model.load_state_dict(state_dict)
    correct_list = []
    loss_list = []
    sss = [7985, 8651]
    input_index = 8651
    for i, batch in enumerate(testloader,0):
        if i == input_index // BS:
            cat = input_index % BS
            the_input = (batch[0][cat], batch[1][cat])
            break
    for _ in range(args.test_run):
        correct, test_loss = test(model, the_input, device, use_noise=True, mean=mean, std=std)
        correct_list.append(correct)
        loss_list.append(test_loss)

    end_time = time.time()
    print(f"training time: {end_time - start_time}")
    # print(correct_list, loss_list)
    print(np.mean(correct_list))
    print(np.mean(loss_list))
    torch.save([correct_list, loss_list], "one_var0.40_9911_8651.pt")