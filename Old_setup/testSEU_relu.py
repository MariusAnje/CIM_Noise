import torch
from torch import nn
import argparse

from utils import relative_diff, SEU_test
from models import SameConv_ReLU as Net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--relu', action='store_true', default=True,
            help='dataset path')
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    args = parser.parse_args()
    num_layers = 3
    kernel_size = 5
    in_channels = 3
    out_channels = 3
    netParams = [num_layers, kernel_size, in_channels, out_channels]
    nData  = 1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
    diffResult = SEU_test(state_dict, Net, dataset, GT, device, netParams)
    torch.save(diffResult, "result_relu.pt")
    for tensor in diffResult:
        print(tensor.mean(), tensor.std())
    #theOutput = model(theInput)
   
