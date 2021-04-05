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

def train_all_one_epoch(mean, std, eta, model, adv_set, BS, trainloader, index, device, optimizer, criterion, scheduler, testloader, update, noise):
    model.train()
    # loader = tqdm(trainloader, leave=False)
    loader = trainloader
    for i, data in enumerate(loader, 0):
        if noise:
            model.set_noise(mean, std)
        else:
            model.clear_noise()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad_()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if update:
            with torch.no_grad():
                outputs = model(inputs + eta * inputs.grad.data.sign())
                _, predicted = torch.max(outputs.data, 1)
                index_slice = index[i*BS: (i+1) * BS]
                adv_set.sens_table[index_slice] =  (predicted != labels).cpu()

    model.clear_noise()
    correct, _ = test_noise_average(mean, std, model, testloader, device)
    num_sample = len(index)
    return correct, num_sample

def train_noise(epochs, mean, std, model, adv_set, BS, device, optimizer, criterion, scheduler, testloader, fname):
    best_correct = 0
    for epoch in range(epochs):
        trainloader, index = adv_set.whole_set(BS)
        correct, num_adv = train_all_one_epoch(mean, std, 0, model, adv_set, BS, trainloader, index, device, optimizer, criterion, scheduler, testloader, update=False, noise=True)
        num_adv = adv_set.sens_table.sum()
        if correct > best_correct:
            best_correct = correct
            # torch.save(model.state_dict(), fname + str(epoch))
            torch.save(model.state_dict(), fname)
        print(f"noise: {correct}, num_adv: {num_adv}")

def train_adv(epochs, first, adv_ep, mean, std, BS, eta, num, model, adv_set, device, optimizer, optimizer_adv, criterion, scheduler, testloader, trainloader, fname):
    """
    epochs: number of total epochs
    first: warmup epochs that trains all
    adv_ep: number of adversarial epochs after each whole epoch 
    """
    best_correct = 0
    for _ in range(min(first, epochs)):
        trainloader, index = adv_set.whole_set(BS)
        correct, _ = train_all_one_epoch(mean, std, eta, model, adv_set, BS, trainloader, index, device, optimizer_adv, criterion, scheduler, testloader, update=True, noise=True)
        num_adv = adv_set.sens_table.sum()
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)
        print(f"noise: {correct}, num_adv: {num_adv}")
    for _ in range(first, epochs, adv_ep+1):
        trainloader, index = adv_set.whole_set(BS)
        correct, _ = train_all_one_epoch(mean, std, eta, model, adv_set, BS, trainloader, index, device, optimizer_adv, criterion, scheduler, testloader, update=True, noise=True)
        num_adv = adv_set.sens_table.sum()
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)
        print(f"noise: {correct}, num_adv: {num_adv}")
        for _ in range(adv_ep):
            trainloader, index = adv_set.dual_draw(num, BS)
            correct_adv, num_sample = train_all_one_epoch(mean, std, eta, model, adv_set, BS, trainloader, index, device, optimizer_adv, criterion, scheduler, testloader, update=False, noise=True)
            if correct_adv > best_correct:
                best_correct = correct_adv
                torch.save(model.state_dict(), fname)
            print(f"adv: {correct_adv}, num_sample: {num_sample}")

def train_adv_ballance(epochs, first, adv_ep, mean, std, BS, eta, num, model, adv_set, device, optimizer, optimizer_adv, criterion, scheduler, testloader, trainloader, fname):
    """
    epochs: number of total epochs
    first: warmup epochs that trains all
    adv_ep: number of adversarial epochs after each whole epoch 
    """
    best_correct = 0
    Total_inf = len(adv_set.sens_table) * epochs
    current_inf = 0
    for _ in range(min(first, epochs)):
        trainloader, index = adv_set.whole_set(BS)
        correct, _ = train_all_one_epoch(mean, std, eta, model, adv_set, BS, trainloader, index, device, optimizer_adv, criterion, scheduler, testloader, update=True, noise=True)
        num_adv = adv_set.sens_table.sum()
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)
        print(f"noise: {correct}, num_adv: {num_adv}")
        current_inf += len(adv_set.sens_table)
    while current_inf <= Total_inf:
        trainloader, index = adv_set.whole_set(BS)
        correct, _ = train_all_one_epoch(mean, std, eta, model, adv_set, BS, trainloader, index, device, optimizer_adv, criterion, scheduler, testloader, update=True, noise=True)
        num_adv = adv_set.sens_table.sum()
        current_inf += len(adv_set.sens_table)
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)
        print(f"noise: {correct}, num_adv: {num_adv}")
        if num_adv < BS // 2:
            break
        for _ in range(adv_ep):
            trainloader, index = adv_set.ballance_draw(num, BS)
            correct_adv, num_sample = train_all_one_epoch(mean, std, eta, model, adv_set, BS, trainloader, index, device, optimizer_adv, criterion, scheduler, testloader, update=False, noise=True)
            current_inf += num_sample
            if correct_adv > best_correct:
                best_correct = correct_adv
                torch.save(model.state_dict(), fname)
            print(f"adv: {correct_adv}, num_sample: {num_sample}")


def train_comb(epochs, first, adv_ep, mean, std, BS, eta, num, model, adv_set, device, optimizer, optimizer_adv, criterion, scheduler, testloader, trainloader, fname):
    """
    epochs: number of total epochs
    first: warmup epochs that trains all
    adv_ep: number of adversarial epochs after each whole epoch 
    """
    best_correct = 0
    for _ in range(min(first, epochs)):
        trainloader, index = adv_set.whole_set(BS)
        correct, num_adv = train_all_one_epoch(mean, std, eta, model, adv_set, BS, trainloader, index, device, optimizer_adv, criterion, scheduler, testloader, update=True, noise=True)
        num_adv = adv_set.sens_table.sum()
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)
        print(f"noise: {correct}, num_adv: {num_adv}")
    for _ in range(first, epochs, adv_ep+1):
        trainloader, index = adv_set.whole_set(BS)
        correct, num_adv = train_all_one_epoch(mean, std, eta, model, adv_set, BS, trainloader, index, device, optimizer_adv, criterion, scheduler, testloader, update=True, noise=True)
        num_adv = adv_set.sens_table.sum()
        if correct > best_correct:
            best_correct = correct
            torch.save(model.state_dict(), fname)
        print(f"noise: {correct}, num_adv: {num_adv}")
        for _ in range(adv_ep):
            trainloader, index = adv_set.sample(num, BS)
            correct_adv, num_sample = train_all_one_epoch(mean, std, eta, model, adv_set, BS, trainloader, index, device, optimizer_adv, criterion, scheduler, testloader, update=False, noise=True)
            if correct_adv > best_correct:
                best_correct = correct_adv
                torch.save(model.state_dict(), fname)
            print(f"adv: {correct_adv}, num_sample: {num_sample}")