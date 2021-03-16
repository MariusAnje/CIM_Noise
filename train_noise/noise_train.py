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
from models import LeNet_Gaussian
from modules import AdvDataset
from utils import *
import argparse
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    parser.add_argument('--BS', action='store', type=int, default=128,
            help='input the batch size')
    parser.add_argument('--fname_head', action='store', default="GCONV_state_dict",
            help='input the filename')
    parser.add_argument('--method', action='store', choices = ["normal", "noise", "adv", "comb"], default="adv", 
            help='input the training method')
    parser.add_argument('--epochs', action='store', type=int, default=5, 
            help='input the number of epochs for training')
    parser.add_argument('--first', action='store', type=int, default=5, 
            help='input the number of first runs for whole training')
    parser.add_argument('--adv_ep', action='store', type=int, default=3, 
            help='input the number of adversarial epochs per iteration')
    parser.add_argument('--adv_num', action='store', type=int, default=10000, 
            help='input the number of samples for adv train')
    parser.add_argument('--test_run', action='store', type=int, default=10, 
            help='input the number of runs for noisy test')
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

    model = LeNet_Gaussian()
    fname_head = args.fname_head + "_" + args.method + "_" + str(time.time())

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer_normal = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    optimizer_noise = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer_noise, 100 * len(trainloader), 0.1)
    adv_set = AdvDataset(trainset)
    mean, std = 0, 0.1

    model.to(device)
    model.clear_noise()
    print(f"method {args.method}")
    print(args)
    start_time = time.time()
    if args.method == "normal":
        train(args.epochs, model, trainloader, device, optimizer_noise, criterion, scheduler, testloader, f"{fname_head}_tmp_best.pt")
    elif args.method == "noise":
        train_noise(args.epochs, mean, std, model, trainloader, device, optimizer_noise, criterion, scheduler, testloader, f"{fname_head}_tmp_best.pt")
    elif args.method == "adv":
        train_adv(args.epochs,  args.first, args.adv_ep, mean, std, BS, 0.1, args.adv_num, model, adv_set, device, optimizer_normal, optimizer_noise, criterion, scheduler, testloader, trainloader, f"{fname_head}_tmp_best.pt")
    elif args.method == "comb":
        train_comb(args.epochs, args.first, args.adv_ep, mean, std, BS, 0.1, args.adv_num, model, adv_set, device, optimizer_normal, optimizer_noise, criterion, scheduler, testloader, trainloader, f"{fname_head}_tmp_best.pt")
    else:
        raise Exception("Not implemented")
    
    end_time = time.time()
    print(f"training time: {end_time - start_time}")
    tmp = torch.load(f"{fname_head}_tmp_best.pt")
    model.load_state_dict(tmp)
    correct, total = test(model, testloader, device)
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (
        100 * correct / total))
    torch.save(model.state_dict(), f"{fname_head}_{correct}.pt")
    clear_c, noisy_list = test_noise(mean, std, args.test_run, model, testloader, device)
    noisy_list = np.array(noisy_list) / 10000
    print(f"mean: {np.mean(noisy_list)}, std: {np.std(noisy_list)}")