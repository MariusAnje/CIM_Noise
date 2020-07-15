import torch
from torch import nn
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
import pickle

"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1  = nn.Linear(784,100)
        self.fc2  = nn.Linear(100,100)
        self.fc3  = nn.Linear(100,10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1  = nn.Linear(2,2)
        self.fc2  = nn.Linear(2,2)
        self.fc3  = nn.Linear(2,2)
        self.fc4  = nn.Linear(2,2)
        self.fc5  = nn.Linear(2,2)
        self.fc6  = nn.Linear(2,2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(self.relu(x))
        # x = self.fc3(self.relu(x))
        # x = self.fc4(self.relu(x))
        # x = self.fc5(self.relu(x))
        # x = self.fc6(self.relu(x))
        x = self.fc6(x)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--relu', action='store_true', default=False,
            help='dataset path')
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    args = parser.parse_args()
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    BS = 128

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False, num_workers=2)

    model = Net()

    """
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    model.to(device)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1,784)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images = images.view(-1,784)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    torch.save(model.state_dict(), "FileName.pt")
    exit(0)
    nData = 1
    dataset  = []

    for i in range(nData):
        theInput = torch.randn(1,100)
        dataset.append(theInput)

    GT = []
    for theInput in dataset:
        theInput  = theInput.to(device)
        theOutput = model(theInput)
        GT.append(theOutput)
    """

    # state_dict = torch.load("FileName.pt", map_location=device)

    oModel = Net()
    state_dict = oModel.state_dict()
    # oModel.load_state_dict(state_dict)
    oModel.to(device)
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        images = images.view(-1,784)
        mean = torch.zeros(1,2)
        std = torch.ones(1,2)
        images = torch.normal(mean,std)
        outputs = oModel(images)
        _, predicted = torch.max(outputs.data, 1)
        GT = outputs.detach().cpu().numpy()
        break
    
    print(GT)


    noise = [(0,1e-2), (0,0), (0,1e-2), (0,0), (0,1e-2), (0,0), (0,1e-2), (0,0), (0,1e-2), (0,0), (0,1e-2), (0,0)]
    pOutput = []
    nRuns = 10000
    import tqdm
    for _ in tqdm.tqdm(range(nRuns)):
    # for _ in range(nRuns):
        if len(state_dict.keys()) != len(noise):
            raise Exception(f"the number of noise doesn't match the number of parameters!\n {len(state_dict.keys())}")
        pModel = Net()
        pModel.load_state_dict(state_dict)
        pState = pModel.state_dict()
        for noise_index, key in enumerate(state_dict.keys()):
            size = state_dict[key].size()
            mean, std = noise[noise_index]
            sampled_noise = torch.randn(size) * std + mean
            pState[key] = pState[key].data + sampled_noise
        pModel.load_state_dict(pState)
        pModel.to(device)
        
        with torch.no_grad():
            for data in testloader:
                # images, labels = data
                # images, labels = images.to(device), labels.to(device)
                # images = images.view(-1,784)
                outputs = pModel(images)
                _, predicted = torch.max(outputs.data, 1)
                pOutput.append(outputs.detach().cpu().numpy())
                break
    
    

    
    for j in range(2):
        shoot = []
        for i in range(len(pOutput)):
            shoot.append(pOutput[i][0][j] - GT[0][j])
        import numpy as np
        print(np.std(shoot))
        plt.hist(shoot,30)
        # plt.savefig(f"fig{j}")
        plt.show()
        f = open(f"2x2_list{j}","wb+")
        pickle.dump(shoot,f)
   
