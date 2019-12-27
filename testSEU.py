import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv6 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

def relative_diff(GT, pOutput):
    if len(GT) != len(pOutput):
        raise Exception("length of GT and Output are different")
    size = GT[0].size()
    totalSize = 1
    for item in size:
        totalSize *= item
    rel = 0
    for i in range(len(GT)):
        rel += ((((GT[i]-pOutput[i])/GT[i]).abs()).sum()/totalSize)
    return rel/len(GT)

if __name__ == "__main__":
    nData  = 16
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

    diffResult = []
    print(len(state_dict))
    for key in state_dict.keys():
        size = state_dict[key].size()
        diffResult_item = torch.zeros(size)
        for i in range(size[0]):
            for o in range(size[1]):
                for r in range(size[2]):
                    for s in range(size[3]):
                        pModel = Net()
                        pModel.load_state_dict(state_dict)
                        pState = pModel.state_dict()
                        pState[key][i,o,r,s].data *= -1
                        pModel.to(device)
                        pOutput = []
                        for theInput in dataset:
                            theInput  = theInput.to(device)
                            theOutput = pModel(theInput)
                            pOutput.append(theOutput)
                        diffResult_item.data[i,o,r,s] = relative_diff(GT,pOutput)
        diffResult.append(diffResult_item)
    
    torch.save(diffResult, "result.pt")
    for tensor in diffResult:
        print(tensor.mean(), tensor.std())
    #theOutput = model(theInput)
   
