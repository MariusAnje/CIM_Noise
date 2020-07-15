import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import pickle
import sklearn

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

def remove_zero(x):
    out = []
    for item in x:
        if item != 0:
            out.append(item)
    return out

# Create models from data
def fit_norm(data, bins=200, the_range=None, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    old_data = data
    new_data = remove_zero(data)
    print(len(data))
    y, x = np.histogram(data, bins=bins, density=True, range=the_range)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    distribution = st.norm

    # fit dist to data
    params = distribution.fit(data)

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Calculate fitted PDF and error with fit in distribution
    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
    
    if len(new_data) != 0:
        plt.bar(x,y, max(data)/len(x),color="#32908F")
        plt.plot(x,pdf,color="#553A41")
        plt.show()
        sse = np.sum(np.power(y - pdf, 2.0))
        kld = st.entropy(y,pdf)/len(y)

        return params, sse, kld
    else:
        return params, 0, 0

# def kld_measure(dist,GT):
#     q = dist/sum(dist)
#     p = GT/sum(GT)
#     return np.sum(np.where(p != 0, p * np.log(p / (q+1e-10)), 0))/len(dist)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

for i in range(10):
    f = open(f"pic1/Conv_list{i}","rb")
    data = pickle.load(f)
    params, sse, kld = fit_norm(data,64, (-1,1))
    print(kld)
