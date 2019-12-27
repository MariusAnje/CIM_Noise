import torch
from torch import nn

def relative_diff(GT, pOutput):
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

def SEU_test(state_dict, Net, dataset, GT, device, netParams):
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
                        diffResult_item.data[i,o,r,s] = relative_diff(GT,pOutput)
        diffResult.append(diffResult_item)
    
    return diffResult