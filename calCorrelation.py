import pickle
import numpy as np

def corr(x1L, x2L):
    x1 = np.array(x1L)
    x2 = np.array(x2L)
    cov = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
    return cov/(x1.std() * x2.std())

if __name__ == "__main__":
    resultList = []
    for i in range(10):
        with open(f"Conv_list{i}", "rb") as inFile:
            resultList.append(pickle.load(inFile))
    
    theMax = 0
    theMin = 2
    corrList = []
    for i in range(10):
        for j in range(10):
            if i < j:
                correlation = corr(resultList[i], resultList[j])
                corrList.append(correlation)
                print(f"{correlation: .2f},", end="")
            else:
                print(f" "*5+",", end="")
        print("")
        
    
    corrList = np.array(corrList)
    print(np.abs(corrList).max(), np.abs(corrList).min(), np.abs(corrList).mean())