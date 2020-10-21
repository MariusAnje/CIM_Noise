import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
# tqdm is imported for better visualization
import tqdm
import modelNeuroSIM
import resnet

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
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                loader.set_description(f"{running_loss/(i+1):.4f}")
        acc = test(device)
        print(f"Epoch {epoch}: test accuracy: {acc:.4f}")
        if acc > best_Acc:
            best_Acc = acc
            torch.save(net.state_dict(), f'./CIFAR10_BN_Aug_{args.model}.pt')

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    parser.add_argument('--batch_size', action='store', type=int, default=256,
            help='input the batch size used in training and inference')
    parser.add_argument('--nruns', action='store', type=int, default=100,
            help='number of runs for test')
    parser.add_argument('--model', action='store', type=str, default='NS', choices=['NS', 'RES'],
            help='which model to use')
    args = parser.parse_args()
    
    # Determining the use scheme
    offline = True

    # Hyper parameters for training offline and inference online
    batchSize = args.batch_size
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # dataset extracting and data preprocessing
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        normalize])
    train_transform = transforms.Compose([
            transforms.RandomAffine(10, translate=(0.07, 0.07)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    trainset = torchvision.datasets.CIFAR10(root='~/Private/data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='~/Private/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=4)

    # model
    if args.model == 'NS':
        net = modelNeuroSIM.NeuroSIM_BN_Model()
    elif args.model == 'RES':
        net = resnet.resnet56_cifar(num_classes=10)
    net.to(device)

    if offline:
        # Offline training
        
        # loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=1e-3)

        # Training
        train(args.nruns, device)

        # Validation
        state_dict = torch.load(f"./CIFAR10_BN_Aug_{args.model}.pt")
        net.load_state_dict(state_dict)
        print(f"Test accuracy: {test(device)}")

    else:
        # Online inference

        # The pretrained model
        state_dict = torch.load(f"./CIFAR10_BN_Aug_{args.model}.pt")
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
