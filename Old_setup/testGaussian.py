import torch
from torch import nn
import argparse

from utils import relative_sum_diff as relative_diff
from utils import gaussian_noise_test
from models import SameConv_ReLU, SameConv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--relu', action='store_true', default=False,
            help='dataset path')
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    args = parser.parse_args()
    num_layers = 7
    kernel_size = 5
    in_channels = 3
    out_channels = 3
    netParams = [num_layers, kernel_size, in_channels, out_channels]
    nData  = 16
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
    noise = [(0,1e-2), (0,1e-2), (0,1e-2), (0,1e-2), (0,1e-2), (0,1e-2), (0,1e-2)]
    diffResult = gaussian_noise_test(state_dict, Net, dataset, GT, noise, device, netParams, relative_diff)
    if args.relu:
        torch.save(diffResult, "result_relu.pt")
    else:
        torch.save(diffResult, "result.pt")
    for tensor in diffResult:
        print(f"mean: {tensor.mean().cpu().numpy():.2f}; std: {tensor.std().cpu().numpy():.2f}; max: {tensor.max().cpu().numpy():.2f}")
        # print(f"mean: {tensor.mean().cpu().numpy()}; std: {tensor.std().cpu().numpy()}; max: {tensor.max().cpu().numpy()}")
    #theOutput = model(theInput)
   
