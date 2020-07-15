import torch
from torch import nn
import argparse
import numpy as np
from matplotlib import pyplot as plt

from utils import relative_sum_diff as relative_diff
from utils import abs_diff
from models import SameConv_ReLU, SameConv

def gaussian_noise_test(state_dict, Net, dataset, GT, noise, device, netParams, metric):
    """
    Pattern of noise:
    [(mean, std), (mean,std), ... , (mean, std)]
    got a mean and a std for each layer
    """
    import tqdm
    diffResult = []
    if len(state_dict.keys()) != len(noise):
        raise Exception("the number of noise doesn't match the number of parameters!")
    for noise_index, key in enumerate(state_dict.keys()):
        size = state_dict[key].size()
        total_number = size[0] * size[1] * size[2] * size[3] * 20
        diffResult_item = torch.zeros(total_number)
        absoluteResult_item = torch.zeros(total_number)
        for i in tqdm.tqdm(range(total_number)):
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
            diffResult_item.data[i] = metric(GT,pOutput)
            absoluteResult_item.data[i] = abs_diff(GT,pOutput)
        diffResult.append((diffResult_item, absoluteResult_item))
    
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
    state_dict = model.state_dict()
    keys = state_dict.keys()
    for key in keys:
        state_dict[key].data = torch.randn(state_dict[key].size()) * 10
    model.load_state_dict(state_dict)
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
    noise = [(0,1)]

    res = gaussian_noise_test(state_dict, Net, dataset, GT, noise, device, netParams, relative_diff)
    # if args.relu:
    #     torch.save(diffResult, "result_relu.pt")
    # else:
    #     torch.save(diffResult, "result.pt")
    for i, item in enumerate(res):
        tensor, absolute = item
        print(f"mean: {tensor.mean().cpu().numpy():.4f}; std: {tensor.std().cpu().numpy():.4f}; max: {tensor.max().cpu().numpy():.4f}")
        print(f"mean: {absolute.mean().cpu().numpy():.4f}; std: {absolute.std().cpu().numpy():.4f}; max: {absolute.max().cpu().numpy():.4f}")
        oneLine = tensor.view(-1)
        absolute = absolute.view(-1)
        torch.save(oneLine,f'relative_{i}.pt')
        torch.save(absolute,f'absolute_{i}.pt')
        oneLine = oneLine.numpy()
        plt.figure()
        plt.hist(oneLine,50)
        plt.savefig(f"relative_{i}")
        plt.figure()
        plt.hist(absolute,50)
        plt.savefig(f"absolute_{i}")

