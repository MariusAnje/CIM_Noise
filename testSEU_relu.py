import torch
from torch import nn
from utils import relative_diff, SEU_test

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
#         self.conv2 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
#         self.conv3 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
#         self.conv4 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
#         self.conv5 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
#         self.conv6 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
#         self.relu  = nn.ReLU()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.conv4(x)
#         x = self.relu(x)
#         x = self.conv5(x)
#         x = self.relu(x)
#         x = self.conv6(x)
#         return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        return x

if __name__ == "__main__":
    nData  = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model    = Net()
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
    diffResult = SEU_test(state_dict, Net, dataset, GT, device)
    torch.save(diffResult, "result_relu.pt")
    for tensor in diffResult:
        print(tensor.mean(), tensor.std())
    #theOutput = model(theInput)
   
