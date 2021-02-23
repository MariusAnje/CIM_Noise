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

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 7 * 7, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()

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

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                            shuffle=True, num_workers=8)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024,
                                            shuffle=False, num_workers=8)

    model = Net()

    """
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    model.to(device)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        loader = tqdm.tqdm(trainloader)
        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.view(-1,784)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loader.set_description(f"{loss:.5f}")
            loss.backward()
            optimizer.step()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1,784)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    torch.save(model.state_dict(), "GCONV_state_dict.pt")
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

    state_dict = torch.load("GCONV_state_dict.pt", map_location=device)

    oModel = Net()
    # state_dict = oModel.state_dict()
    oModel.load_state_dict(state_dict)
    oModel.to(device)
    correct = 0
    total   = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # images = images.view(-1,784)
        # mean = torch.zeros(1,2)
        # std = torch.ones(1,2)
        # images = torch.normal(mean,std)
        outputs = oModel(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        total   += len(predicted)
    
    acc_filename = correct
    print(f"{correct}/{total} = {correct*1./total:.4f}")


    noise = (0,5e-2)
    pOutput = []
    nRuns = 10000
    acc = []
    import tqdm
    for _ in tqdm.tqdm(range(nRuns)):
    # for _ in range(nRuns):
        pModel = Net()
        pModel.load_state_dict(state_dict)
        pState = pModel.state_dict()
        for noise_index, key in enumerate(state_dict.keys()):
            if key.find("weight") != -1:
                size = state_dict[key].size()
                mean, std = noise
                sampled_noise = torch.randn(size) * std + mean
                pState[key] = pState[key].data + sampled_noise
        pModel.load_state_dict(pState)
        pModel.to(device)
        
        correct = 0
        total   = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                # images = images.view(-1,784)
                outputs = pModel(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum()
                total   += len(predicted)
            acc.append((correct).detach().cpu().numpy())
    
    

    # plt.hist(acc,30)
    # plt.savefig(f"fig{j}")
    # plt.show()
    # f = open(f"acc9874","wb+")
    # pickle.dump(acc,f)
    torch.save(acc, f"acc_{acc_filename}_{noise[1]}_{nRuns}")
   
