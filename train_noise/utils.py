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

def test_noise_average(mean, std, model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            model.set_noise(mean, std)
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1,784)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.clear_noise()
    return correct, total


def train_noise_one_epoch(mean, std, model, trainloader, device, optimizer, criterion, scheduler, testloader):
    model.train()
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
    correct, _ = test_noise_average(mean, std, model, testloader, device)
    print(correct)
    return correct

def train_noise(epochs, mean, std, model, trainloader, device, optimizer, criterion, scheduler, testloader, fname):
    best_correct = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        correct = train_noise_one_epoch(mean, std, model, trainloader, device, optimizer, criterion, scheduler, testloader)
        if correct > best_correct:
            best_correct = correct
            # torch.save(model.state_dict(), fname + str(epoch))
            torch.save(model.state_dict(), fname)

def test_adv_one_epoch(mean, std, BS, eta, model, adv_set, device, optimizer, criterion, scheduler, testloader):
    model.train()
    data_loader, index = adv_set.whole_set(BS)
    # loader = tqdm(data_loader, leave=False)
    loader = data_loader
    for i, data in enumerate(loader, 0):
        model.clear_noise()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        with torch.no_grad():
            outputs = model(inputs + eta * inputs)
            _, predicted = torch.max(outputs.data, 1)
            index_slice = index[i*BS: (i+1) * BS]
            adv_set.sens_table[index_slice] =  (predicted != labels).cpu()
        
    correct, _ = test(model, testloader, device)
    correct = 0
    return correct


def train_adv_one_epoch(mean, std, BS, eta, model, adv_set, device, optimizer, criterion, scheduler, testloader):
    model.train()
    data_loader, index = adv_set.whole_set(BS)
    # loader = tqdm(data_loader, leave=False)
    loader = data_loader
    for i, data in enumerate(loader, 0):
        # model.set_noise(mean, std)
        model.clear_noise()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # optimizer.step()
        
        with torch.no_grad():
            outputs = model(inputs + eta * inputs)
            _, predicted = torch.max(outputs.data, 1)
            index_slice = index[i*BS: (i+1) * BS]
            adv_set.sens_table[index_slice] =  (predicted != labels).cpu()

    model.clear_noise()
    correct, _ = test_noise_average(mean, std, model, testloader, device)
    num_adv = adv_set.sens_table.sum()
    return correct, num_adv

def train_adv_sens_set(mean, std, BS, num, model, adv_set, device, optimizer, criterion, scheduler, testloader):
    model.train()
    running_loss = 0.0
    # loader = tqdm(adv_set.sample(num, BS), leave=False)
    # loader, num_sample = adv_set.sample(num, BS)
    loader, num_sample = adv_set.dual_draw(num, BS)
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
    correct, _ = test_noise_average(mean, std, model, testloader, device)
    # correct, _ = test(model, testloader, device)
    return correct, num_sample

def train_adv(epochs, first, adv_ep, mean, std, BS, eta, num, model, adv_set, device, optimizer, optimizer_adv, criterion, scheduler, testloader, trainloader, fname):
    """
    epochs: number of total epochs
    first: warmup epochs that trains all
    adv_ep: number of adversarial epochs after each whole epoch 
    """
    best_correct = 0
    for _ in range(min(first, epochs)):
        # correct, num_adv = train_adv_one_epoch(mean, std, BS, eta, model, adv_set, device, optimizer, criterion, scheduler, testloader)
        correct, num_adv = train_comb_one_epoch(mean, std, BS, eta, model, adv_set, device, optimizer_adv, criterion, scheduler, testloader)
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)
        scheduler.step()
        print(f"noise: {correct}, num_adv: {num_adv}")
    for _ in range(first, epochs, adv_ep+1):
        correct, num_adv = train_adv_one_epoch(mean, std, BS, eta, model, adv_set, device, optimizer, criterion, scheduler, testloader)
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)
        print(f"normal: {correct}, num_adv: {num_adv}")
        for _ in range(adv_ep):
            correct_adv, num_sample = train_adv_sens_set(mean, std, BS, num, model, adv_set, device, optimizer_adv, criterion, scheduler, testloader)
            if correct_adv > best_correct:
                best_correct = correct_adv
                torch.save(model.state_dict(), fname)
            print(f"adv: {correct_adv}, num_sample: {num_sample}")
            
            scheduler.step()

def train_comb_one_epoch(mean, std, BS, eta, model, adv_set, device, optimizer, criterion, scheduler, testloader):
    model.train()
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
        scheduler.step()
        
        with torch.no_grad():
            outputs = model(inputs + eta * inputs)
            _, predicted = torch.max(outputs.data, 1)
            index_slice = index[i*BS: (i+1) * BS]
            adv_set.sens_table[index_slice] =  (predicted != labels).cpu()

    model.clear_noise()
    correct, _ = test_noise_average(mean, std, model, testloader, device)
    num_adv = adv_set.sens_table.sum()
    return correct, num_adv

def train_comb_sens_set(mean, std, BS, num, model, adv_set, device, optimizer, criterion, scheduler, testloader):
    model.train()
    running_loss = 0.0
    # loader = tqdm(adv_set.sample(num, BS), leave=False)
    # loader = adv_set.sample(num, BS)
    loader = adv_set.dual_draw(num, BS)
    # loader = adv_set.whole_set(BS)[0]
    for _, data in enumerate(loader, 0):
        model.set_noise(mean, std)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    model.clear_noise()
    correct, _ = test_noise_average(mean, std, model, testloader, device)
    return correct

def train_comb(epochs, first, adv_ep, mean, std, BS, eta, num, model, adv_set, device, optimizer, optimizer_adv, criterion, scheduler, testloader, trainloader, fname):
    """
    epochs: number of total epochs
    first: warmup epochs that trains all
    adv_ep: number of adversarial epochs after each whole epoch 
    """
    best_correct = 0
    for _ in range(min(first, epochs)):
        correct = train_comb_one_epoch(mean, std, BS, eta, model, adv_set, device, optimizer, criterion, scheduler, testloader)
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)
        scheduler.step()
        print(f"normal: {correct}")
    for _ in range(first, epochs, adv_ep+1):
        for _ in range(adv_ep):
            correct_adv = train_comb_sens_set(mean, std, BS, num, model, adv_set, device, optimizer, criterion, scheduler, testloader)
            print(f"adv: {correct_adv}")
        correct = train_comb_one_epoch(mean, std, BS, eta, model, adv_set, device, optimizer, criterion, scheduler, testloader)

        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)
        print(f"normal: {correct}")
        scheduler.step()