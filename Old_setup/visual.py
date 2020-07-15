import torch
import numpy as np
from matplotlib import pyplot as plt

fileName = "result_relu.pt"
theList = torch.load(fileName)

for tensor in theList:
    oneLine = tensor.view(-1)
    oneLine = oneLine.numpy()
    plt.hist(oneLine,25)
    plt.show()
