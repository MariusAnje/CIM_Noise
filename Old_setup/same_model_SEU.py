import torch
from torch import nn
import argparse

from utils import relative_single_diff, relative_sum_diff, SEU_test, copy_state_dict
from models import SameConv_ReLU, SameConv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    parser.add_argument('--metric', action='store', default='single',
            help='input the metric you use: single or sum')
    args = parser.parse_args()
    num_layers = 6
    kernel_size = 5
    in_channels = 3
    out_channels = 3
    netParams = [num_layers, kernel_size, in_channels, out_channels]
    nData  = 16
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if   args.metric == "single":
        relative_diff = relative_single_diff
    elif args.metric == "sum":
        relative_diff = relative_sum_diff

    dataset  = []
    for i in range(nData):
        theInput = torch.randn(256,3,100,100)
        dataset.append(theInput)

    
    Net   = SameConv
    model = Net(num_layers, kernel_size, in_channels, out_channels)
    model.to(device)
    state_dict = model.state_dict()

    
    GT = []
    for theInput in dataset:
        theInput  = theInput.to(device)
        theOutput = model(theInput)
        GT.append(theOutput)
        
    diffResult = SEU_test(state_dict, Net, dataset, GT, device, netParams, relative_diff)
    torch.save(diffResult, "result_same.pt")
    print("Without ReLU:")
    for tensor in diffResult:
        print(f"mean: {tensor.mean().cpu().numpy():.3f}; std: {tensor.std().cpu().numpy():.2f}; max: {tensor.max().cpu().numpy():.2f}")
    #theOutput = model(theInput)

    Net = SameConv_ReLU
    model = Net(num_layers, kernel_size, in_channels, out_channels)
    model.to(device)
    state_dict = copy_state_dict(state_dict, model.state_dict())
    model.load_state_dict(state_dict)
    GT = []
    for theInput in dataset:
        theInput  = theInput.to(device)
        theOutput = model(theInput)
        GT.append(theOutput)
        
    diffResult = SEU_test(state_dict, Net, dataset, GT, device, netParams, relative_diff)
    torch.save(diffResult, "result_relu_same.pt")
    print("\nWith ReLU:")
    for tensor in diffResult:
        print(f"mean: {tensor.mean().cpu().numpy():.3f}; std: {tensor.std().cpu().numpy():.2f}; max: {tensor.max().cpu().numpy():.2f}")
   
