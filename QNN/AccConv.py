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
import modelNeuroSIM
import resnet
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    parser.add_argument('--batch_size', action='store', type=int, default=256,
            help='input the batch size used in training and inference')
    parser.add_argument('--nruns', action='store', type=int, default=100,
            help='number of runs for test')
    parser.add_argument('--model', action='store', type=str, default='NS', choices=['NS', 'RES'],
            help='which model to use')
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    BS = args.batch_size
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
#     normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose(
    [transforms.ToTensor(),
     normalize])
    # trainset = torchvision.datasets.CIFAR10(root='~/Private/data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='~/Private/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=4)
    
    if args.model == "NS":
        model = modelNeuroSIM.NeuroSIM_BN_Model()
    elif args.model == "RES":
        model = resnet.resnet56_cifar(num_classes=10)

    model = modelNeuroSIM.NeuroSIM_BN_Model()

    state_dict = torch.load(f"CIFAR10_BN_Aug_{args.model}.pt", map_location=device)

    if args.model == "NS":
        oModel = modelNeuroSIM.NeuroSIM_BN_Model()
    elif args.model == "RES":
        oModel = resnet.resnet56_cifar(num_classes=10)

    oModel.load_state_dict(state_dict)
    oModel.to(device)
    oModel.eval()
    correct = 0
    total   = 0
#     for data in tqdm.tqdm(testloader):
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = oModel(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        total   += len(predicted)
    
    oAcc = correct*1./total
    print(f"{correct}/{total} = {correct*1./total:.4f}")


    noise = (0,5e-2)
    pOutput = []
    nRuns = args.nruns
    pAcc = []
    import tqdm
#     for _ in tqdm.tqdm(range(nRuns)):
    for _ in range(nRuns):
        if args.model == "NS":
            pModel = modelNeuroSIM.NeuroSIM_BN_Model()
        elif args.model == "RES":
            pModel = resnet.resnet56_cifar(num_classes=10)

        pModel.load_state_dict(state_dict)
        pState = pModel.state_dict()
        for noise_index, key in enumerate(state_dict.keys()):
            if key.find("weight") != -1 and key.find("conv") != -1:
                size = state_dict[key].size()
                mean, std = noise
                sampled_noise = torch.randn(size) * std + mean
                pState[key] = pState[key].data + sampled_noise
        pModel.load_state_dict(pState)
        pModel.to(device)
        
        correct = 0
        total   = 0
        with torch.no_grad():
            pModel.eval()
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                # images = images.view(-1,784)
                outputs = pModel(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum()
                total   += len(predicted)
            pAcc.append((correct).detach().cpu().item())
    if args.nruns < 100:
        print(pAcc)
    torch.save(pAcc, f"acc_vari_{args.model}_{int(oAcc*10000)}_{time.time()}")
