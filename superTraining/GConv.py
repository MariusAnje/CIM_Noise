import torch
from torch import nn
import torch.nn.functional as F
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

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


def train(numEpoch, device):
    """
        Typical training scheme for offline training.
    """
    net.train()
    best_Acc = 0.0
    for epoch in range(numEpoch):  # loop over the dataset multiple times

        running_loss = 0.0
        with tqdm.tqdm(trainloader, leave = False) as loader:
            for i, data in enumerate(loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.data += 10
                loss.backward()
                optimizer.step()
                loss.data -= 10

                # print statistics
                running_loss += loss.item()
                loader.set_description(f"{running_loss/(i+1):.4f}")
        acc = test(device)
        print(f"Epoch {epoch}: test accuracy: {acc:.4f}, loss: {running_loss:.4f}")
        if acc > best_Acc:
            best_Acc = acc
            torch.save(net.state_dict(), './weightSuper.pt')

def test(device):
    """
        Typical inference scheme for both offline validation and online inference.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total


if __name__ == "__main__":
    
    # Determining the use scheme
    offline = True

    # Hyper parameters for training offline and inference online
    batchSize = 128
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # dataset extracting and data preprocessing
    transform = transforms.Compose(
    [transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='~/Private/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=4)

    # model
    net = Net()
    net.to(device)

    if offline:
        # Offline training
        
        # loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.1)
        # optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)

        # Training
        train(20, device)

        # Validation
        state_dict = torch.load("./weightSuper.pt")
        net.load_state_dict(state_dict)
        print(f"Test accuracy: {test(device)}")

    else:
        # Online inference

        # The pretrained model
        state_dict = torch.load("./weightSuper.pt")
        net.load_state_dict(state_dict)

        # Actual inference
        print(f"Test accuracy: {test(device)}")
        exit()
        print_model = modelNeuroSIM.Visual_Model()
        print_model.to(device)
        print_model.load_state_dict(state_dict)
        with torch.no_grad():
            print_model.eval()
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = print_model(images)
                break