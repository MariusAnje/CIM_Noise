import torch
from torch import nn
import argparse
import numpy as np
from matplotlib import pyplot as plt

from utils import relative_sum_diff as relative_diff
from models import SameConv_ReLU, SameConv

def gaussian_noise_test(state_dict, Net, dataset, GT, noise, device, netParams, metric):
    """
    Pattern of noise:
    [(mean, std), (mean,std), ... , (mean, std)]
    got a mean and a std for each layer
    """
    diffResult = []
    if len(state_dict.keys()) != len(noise):
        raise Exception("the number of noise doesn't match the number of parameters!")
    for noise_index, key in enumerate(state_dict.keys()):
        size = state_dict[key].size()
        diffResult_item = torch.zeros(size)
        for i in range(size[0]):
            for o in range(size[1]):
                for r in range(size[2]):
                    for s in range(size[3]):
                        pModel = Net(netParams[0], netParams[1], netParams[2], netParams[3])
                        pModel.load_state_dict(state_dict)
                        pState = pModel.state_dict()

                        mean, std = noise[noise_index]
                        sampled_noise = torch.randn(size) * std + mean
                        pState[key] = pState[key].data + sampled_noise
                        pModel.load_state_dict(pState)
                        pModel.to(device)
                        pOutput = []
                        for theInput in dataset:
                            theInput  = theInput.to(device)
                            theOutput = pModel(theInput)
                            pOutput.append(theOutput)
                        diffResult_item.data[i,o,r,s] = metric(GT,pOutput)
        diffResult.append(diffResult_item)
    
    return diffResult

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--relu', action='store_true', default=False,
            help='dataset path')
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    args = parser.parse_args()
    num_layers = 1
    kernel_size = 5
    in_channels = 3
    out_channels = 3
    netParams = [num_layers, kernel_size, in_channels, out_channels]
    nData  = 1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.relu:
        Net = SameConv_ReLU
    else:
        Net = SameConv

    model    = Net(num_layers, kernel_size, in_channels, out_channels)
    model.to(device)
    dataset  = []

    for i in range(nData):
        theInput = torch.randn(256,3,100,100)
        dataset.append(theInput)

    GT = []
    for theInput in dataset:
        theInput  = theInput.to(device)
        theOutput = model(theInput)
        GT.append(theOutput)
        

    state_dict = model.state_dict()
    noise = [(0,1e-2)] #, (0,1e-2), (0,1e-2), (0,1e-2), (0,1e-2), (0,1e-2), (0,1e-2)]
    diffResult = gaussian_noise_test(state_dict, Net, dataset, GT, noise, device, netParams, relative_diff)
    # if args.relu:
    #     torch.save(diffResult, "result_relu.pt")
    # else:
    #     torch.save(diffResult, "result.pt")
    for tensor in diffResult:
        print(f"mean: {tensor.mean().cpu().numpy():.2f}; std: {tensor.std().cpu().numpy():.2f}; max: {tensor.max().cpu().numpy():.2f}")
        oneLine = tensor.view(-1)
        torch.save(oneLine,'oneline.pt')
        oneLine = oneLine.numpy()
        plt.plot(oneLine)
        plt.show()

