import torch
from torch import nn
from utils import relative_diff, SEU_test
from models import SameConv_ReLU as Net

if __name__ == "__main__":
    num_layers = 3
    kernel_size = 5
    in_channels = 3
    out_channels = 3
    netParams = [num_layers, kernel_size, in_channels, out_channels]
    nData  = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
   
