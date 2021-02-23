import numpy as np
from matplotlib import pyplot as plt
import torch
import os
from tqdm import tqdm

# path = "C:/Users/Mariu/Desktop/"
path = "./"
# GT = np.load("GT.npy")
GT = np.load(os.path.join(path, "CIFAR10GT.npy"))
CR = np.load(os.path.join(path, f"Res110_{0}_00.npz"))["arr_0"]
print(CR.shape)
CR = np.argmax(CR[0], axis=1)
print(CR.shape)
# exit()


patterns = ["0", "1.0", "5e-1", "1e-1", "4e-2", "1e-2"]
# patterns = ["1.0", "1e-2"]

for pattern in patterns:
    # pattern = "1.0"
    a = np.load(os.path.join(path, f"Res110_{pattern}_00.npz"))["arr_0"]
    # for i in tqdm(range(1,10)):
    #     a_c = np.load(os.path.join(path, f"output4e-2_{i:02d}.npz"))["arr_0"]
    #     a = np.concatenate([a, a_c])
    print(len(a))
    print("done loading data")
    acc_list = []
    css_list = []
    for i in range(len(a)):
        exp = np.argmax(a[i], axis=1)
        acc = (exp == GT).sum()
        css = (exp == CR).sum()
        acc_list.append(acc)
        css_list.append(css)
    print(f"Avg Acc: {np.mean(acc_list)}")
    print(f"Avg Css: {np.mean(css_list)}")
    
    # continue

    a = torch.Tensor(a)
    a = a.softmax(axis=2)
    a = a.numpy()

    epistemics = np.zeros([a.shape[1], a.shape[2]])
    aleatorics = np.zeros([a.shape[1], a.shape[2]])
    for i in tqdm(range(a.shape[1])):
        p_hat = a[:,i, :]
        p_bar = np.mean(p_hat, axis=0)

        temp = p_hat - np.expand_dims(p_bar, 0)
        epistemic = np.dot(temp.T, temp) / a.shape[0]
        epistemic = np.diag(epistemic)
        epistemics[i] = epistemic

        aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / a.shape[0])
        aleatoric = np.diag(aleatoric)
        aleatorics[i] = aleatoric

    # np.save(f"epistemics{pattern}", epistemics)
    # np.save(f"aleatorics{pattern}", aleatorics)
    var = (epistemics + aleatorics)
    care = np.zeros(len(GT))
    for i in range(len(GT)):
        # print(GT[i])
        care[i] = epistemics[i,int(GT[i])]
        # care[i] = var[i].max()

    print(f"max: {care.max()}")
    print(f"min: {care.min()}")
    print(f"mean: {care.mean()}")
    print(f"std: {care.std()}")
