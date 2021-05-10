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
from torch import autograd

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
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1,784)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            running_loss += criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        model.clear_noise()
        return correct/total, (running_loss/total).item()

def train(epochs, model, trainloader, testloader, device, filename):
    start_time = time.time()
    best_correct = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        # loader = tqdm(trainloader, leave=False)
        loader = trainloader
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
            torch.save(model.state_dict(), f"{filename}_best_{start_time}.pt")
    return best_correct, start_time

def train_second_order(epochs, model, trainloader, testloader, device, filename):
    start_time = time.time()
    best_correct = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        # loader = tqdm(trainloader, leave=False)
        loader = trainloader
        for _, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            second_derivative = 0
            for name, w in model.named_parameters():
                if name.find("weight") != -1:
                    # first_derivative = autograd.grad(loss, w, create_graph=True)[0]
                    # for index, f in enumerate(first_derivative.view(-1)):
                    #     second_derivative += autograd.grad(f, w,retain_graph=True)[0].view(-1)[index]
                    
                    first_derivative = autograd.grad(loss, w, create_graph=True)[0].sum()
                    second_derivative += autograd.grad(first_derivative, w,retain_graph=True)[0].view(-1).sum()
            optimizer.zero_grad()
            alpha = 0.2
            loss = alpha * loss + (1 - alpha) * second_derivative
            loss.backward()
            optimizer.step()
            scheduler.step()
        correct, test_loss = test(model, testloader, device, use_noise=False)
        print(f"acc: {correct:.4f}, loss: {test_loss:.4f}")
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), f"{filename}_best_{start_time}.pt")
    return best_correct, start_time


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
    parser.add_argument('--method', action='store', choices = ["normal", "second"], default="normal", 
            help='input the training method')
    parser.add_argument('--epochs', action='store', type=int, default=80, 
            help='# of epochs for training')
    parser.add_argument('--fnhead', action='store', default="GCONV", 
            help='filename head')
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
        if args.method == "normal":
            best_correct, time_stamp = train(args.epochs, model, trainloader, testloader, device, filename=args.fnhead)
        if args.method == "second":
            best_correct, time_stamp = train_second_order(args.epochs, model, trainloader, testloader, device, filename=args.fnhead)
        state_dict = torch.load(f"{args.fnhead}_best_{time_stamp}.pt")
        torch.save(state_dict, f"{args.fnhead}_{best_correct}.pt")
    else:
        best_correct = 9911

    state_dict = torch.load(f"{args.fnhead}_{best_correct}.pt")
    model.load_state_dict(state_dict)
    correct_list = []
    loss_list = []
    for _ in range(args.test_run):
        correct, test_loss = test(model, testloader, device, use_noise=True, mean=mean, std=std)
        correct_list.append(correct)
        loss_list.append(test_loss)

    end_time = time.time()
    print(f"training time: {end_time - start_time}")
    # print(correct_list, loss_list)
    print(np.mean(correct_list))
    print(np.mean(loss_list))
    torch.save([correct_list, loss_list], f"res_var0.40_{best_correct}.pt")