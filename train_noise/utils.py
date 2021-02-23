import torch
from tqdm import tqdm

def train(epochs, model, trainloader, device, optimizer, criterion, scheduler, testloader, fname):
    best_correct = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        # loader = tqdm(trainloader, leave=False)
        loader = trainloader
        for _, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.view(-1,784)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # loader.set_description(f"{loss:.5f}")
            loss.backward()
            optimizer.step()
        scheduler.step()
        correct, _ = test(model, testloader, device)
        print(correct)
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)

def test(model, testloader, device):
    model.eval()
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
    return correct, total

def test_noise(mean, std, nRuns, model, testloader, device):
    model.clear_noise()
    clear_c, total = test(model, testloader, device)
    noisy_list = []
    # for _ in tqdm(range(nRuns)):
    for _ in range(nRuns):
        model.set_noise(mean, std)
        nosiy_c, total = test(model, testloader, device)
        noisy_list.append(nosiy_c)
    return clear_c, noisy_list

def train_noise(epochs, mean, std, model, trainloader, device, optimizer, criterion, scheduler, testloader, fname):
    best_correct = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        # loader = tqdm(trainloader, leave=False)
        loader = trainloader
        for _, data in enumerate(loader, 0):
            model.set_noise(mean, std)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.clear_noise()
        correct, _ = test(model, testloader, device)
        print(correct)
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)

def train_adv_one_epoch(mean, std, BS, eta, model, adv_set, device, optimizer, criterion, scheduler, testloader):
    model.train()
    running_loss = 0.0
    data_loader, index = adv_set.whole_set(BS)
    # loader = tqdm(data_loader, leave=False)
    loader = data_loader
    for i, data in enumerate(loader, 0):
        model.set_noise(mean, std)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            outputs = model(inputs + eta * inputs)
            _, predicted = torch.max(outputs.data, 1)
            index_slice = index[i*BS: (i+1) * BS]
            adv_set.sens_table[index_slice] =  (predicted == labels).cpu()

    scheduler.step()
    model.clear_noise()
    correct, _ = test(model, testloader, device)
    return correct

def train_adv_sens_set(mean, std, BS, num, model, adv_set, device, optimizer, criterion, scheduler, testloader):
    model.train()
    running_loss = 0.0
    # loader = tqdm(adv_set.sample(num, BS), leave=False)
    loader = adv_set.sample(num, BS)
    for _, data in enumerate(loader, 0):
        model.set_noise(mean, std)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
    model.clear_noise()
    correct, _ = test(model, testloader, device)
    return correct

def train_adv(epochs, mean, std, BS, eta, num, model, adv_set, device, optimizer, criterion, scheduler, testloader, fname):
    best_correct = 0
    for _ in range(epochs):
        correct = train_adv_one_epoch(mean, std, BS, eta, model, adv_set, device, optimizer, criterion, scheduler, testloader)
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)
        correct_adv = train_adv_sens_set(mean, std, BS, num, model, adv_set, device, optimizer, criterion, scheduler, testloader)
        print(f"normal: {correct}, adv: {correct_adv}")