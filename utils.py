import torch
from torch import nn

def abs_diff(GT, pOutput):
    eps = pow(2,-16)
    if len(GT) != len(pOutput):
        raise Exception("length of GT and Output are different")
    size = GT[0].size()
    totalSize = 1
    for item in size:
        totalSize *= item
    rel = 0
    for i in range(len(GT)):
        rel += (GT[i]-pOutput[i]).abs().sum()/ totalSize
    return rel/len(GT)

def relative_single_diff(GT, pOutput):
    eps = pow(2,-16)
    if len(GT) != len(pOutput):
        raise Exception("length of GT and Output are different")
    size = GT[0].size()
    totalSize = 1
    for item in size:
        totalSize *= item
    rel = 0
    for i in range(len(GT)):
        rel += ((((GT[i]-pOutput[i])/(GT[i]+eps)).abs()).sum()/totalSize)
    return rel/len(GT)

def relative_sum_diff(GT, pOutput):
    eps = pow(2,-16)
    if len(GT) != len(pOutput):
        raise Exception("length of GT and Output are different")
    size = GT[0].size()
    totalSize = 1
    for item in size:
        totalSize *= item
    rel = 0
    for i in range(len(GT)):
        rel += 2 * (GT[i]-pOutput[i]).abs().sum()/(GT[i].abs()+pOutput[i].abs()).sum()
        # rel += ((((GT[i]-pOutput[i])/(GT[i]+eps)).abs()).sum()/totalSize)
    return rel/len(GT)

def copy_state_dict(state_dict1, state_dict2):
    keys1 = list(state_dict1.keys())
    keys2 = list(state_dict2.keys())

    for i in range(len(keys1)):
        state_dict2[keys2[i]] = state_dict1[keys1[i]]
    
    return state_dict2

def SEU_test(state_dict, Net, dataset, GT, device, netParams, metric):
    diffResult = []
    # print(len(state_dict))
    for key in state_dict.keys():
        size = state_dict[key].size()
        diffResult_item = torch.zeros(size)
        for i in range(size[0]):
            for o in range(size[1]):
                for r in range(size[2]):
                    for s in range(size[3]):
                        pModel = Net(netParams[0], netParams[1], netParams[2], netParams[3])
                        pModel.load_state_dict(state_dict)
                        pState = pModel.state_dict()
                        pState[key][i,o,r,s].data *= -1
                        pModel.to(device)
                        pOutput = []
                        for theInput in dataset:
                            theInput  = theInput.to(device)
                            theOutput = pModel(theInput)
                            pOutput.append(theOutput)
                        diffResult_item.data[i,o,r,s] = metric(GT,pOutput)
        diffResult.append(diffResult_item)
    
    return diffResult

def gaussian_noise_test(state_dict, Net, dataset, GT, noise, device, netParams, metric):
    """
    Pattern of noise:
    [(mean, std), (mean,std), ... , (mean, std)]
    got a mean and a std for each layer
    """
    diffResult = []
    if len(state_dict.keys()) != len(noise):
        raise Exception("the number of noise doesn't match the number of parameters!")
    for noise_index, key in enumerate(state_dict.keys()):
        size = state_dict[key].size()
        diffResult_item = torch.zeros(size)
        for i in range(size[0]):
            for o in range(size[1]):
                for r in range(size[2]):
                    for s in range(size[3]):
                        pModel = Net(netParams[0], netParams[1], netParams[2], netParams[3])
                        pModel.load_state_dict(state_dict)
                        pState = pModel.state_dict()

                        mean, std = noise[noise_index]
                        sampled_noise = torch.randn(size) * std + mean
                        pState[key] = pState[key].data + sampled_noise
                        pModel.load_state_dict(pState)
                        pModel.to(device)
                        pOutput = []
                        for theInput in dataset:
                            theInput  = theInput.to(device)
                            theOutput = pModel(theInput)
                            pOutput.append(theOutput)
                        diffResult_item.data[i,o,r,s] = metric(GT,pOutput)
        diffResult.append(diffResult_item)
    
    return diffResult