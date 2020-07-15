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


net = modelNeuroSIM.NeuroSIM_BN_Model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--relu', action='store_true', default=False,
            help='dataset path')
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    BS = 256

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='~/Private/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='~/Private/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    model = modelNeuroSIM.NeuroSIM_BN_Model()

    state_dict = torch.load("CIFAR10_BN.pt", map_location=device)

    oModel = modelNeuroSIM.NeuroSIM_BN_Model()
    # state_dict = oModel.state_dict()
    oModel.load_state_dict(state_dict)
    oModel.to(device)
    count = 0
    for data in testloader:
        count += 1
        if count == 1232:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1,784)
            # mean = torch.zeros(1,2)
            # std = torch.ones(1,2)
            # images = torch.normal(mean,std)
            outputs = oModel(images)
            _, predicted = torch.max(outputs.data, 1)
            GT = outputs.detach().cpu().numpy()
            break
    
    print(GT)
    # exit(0)


    noise = (0,0.25)
    pOutput = []
    nRuns = 10000
    import tqdm
    # for _ in tqdm.tqdm(range(nRuns)):
    for runTime in range(nRuns):
        logInterval = 100
        if runTime % logInterval == logInterval - 1:
            logFile = open("logFile","a+")
            logFile.write(str(runTime)+"\n")
            logFile.close()
        pModel = modelNeuroSIM.NeuroSIM_BN_Model()
        pModel.load_state_dict(state_dict)
        pState = pModel.state_dict()
        for noise_index, key in enumerate(state_dict.keys()):
            # print(key)
            if key.find("conv.weight") != -1:
                size = state_dict[key].size()
                mean, std = noise
                std = pState[key].data.max() /16
                sampled_noise = torch.randn(size) * std + mean
                pState[key] = pState[key].data + sampled_noise
        pModel.load_state_dict(pState)
        pModel.to(device)
        # exit()
        
        with torch.no_grad():
            for data in testloader:
                # images, labels = data
                # images, labels = images.to(device), labels.to(device)
                # images = images.view(-1,784)
                outputs = pModel(images)
                _, predicted = torch.max(outputs.data, 1)
                pOutput.append(outputs.detach().cpu().numpy())
                break
    
    

    flag = True
    for j in range(10):
        shoot = []
        for i in range(len(pOutput)):
            shoot.append(pOutput[i][0][j] - GT[0][j])
        import numpy as np
        print(np.std(shoot))
        if flag and np.std(shoot) > 1e-4:
            plt.hist(shoot,30)
            # plt.savefig(f"fig{j}")
            plt.show()
            flag = False
        f = open(f"Conv_list{j}","wb+")
        pickle.dump(shoot,f)
   
