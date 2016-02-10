# coding: utf-8

"""fct.py
    
    Here are the general functions that are used through the different 
    notebooks. 

    Can be found here
    - functions to generate datas
    - functions to normalize
    - plotting functions
    
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
import matplotlib.cm as cmx
import matplotlib.colors as colors

## -------------------- Generate datas  -------------------

def generate_multivariate(size=500, dimension=3, mean=0):
    """Returns random variables generated from a multivariate normal."""

    # generate a covariance matrix from a random matrix of size dimension ** 2
    random_matrix = np.random.rand(dimension, dimension)
    
    # a covariance matrix should be semi-definite positive. Such matrix can
    # be generated as the product of a random matrix and its transposed.
    cov = np.dot(random_matrix, random_matrix.T)
    
    # mean is fixed to zero if not specified
    if mean == 0:
        mean = [0] * dimension

    return np.random.multivariate_normal(mean, cov, size)


## -------------------- Normalization  -------------------

def normalize_min_max(data, dimension):
    """Takes a set of data and returns this same set normalized to change the
    range to [0, 1]."""

    for i in range(dimension):
        data[:, i] = ((data[:, i] - np.amin(data[:, i])) / 
                      (np.amax(data[:, i]) - np.amin(data[:, i])))

def normalize(data):
    """Normalize a dataset column by column. Note that it makes a copy and 
    thus might be slower and memory costly than other solutions."""

    res = np.copy(data)
    for i in range(data.shape[1]):
        res[:, i] = ((res[:, i] - np.mean(res[:, i])) / np.std(res[:, i]))
    return res

def normalize_pd(data):
    """Takes a pandas dataframe and returns the latter normalized."""

    return (data - data.mean()) / data.std()


## -------------------- Plotting  -------------------

def plt_3d(data, title='data'):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    data_t = data.T
    ax.scatter(data_t[0], data_t[1], data_t[2], alpha=0.2)
    plt.title(title)

def plot_2d(datas, color='r'):
    datas_t = np.array(datas)
    plt.scatter(datas_t[:,0], datas_t[:,1], alpha=0.2, c=color)
    plt.title('Representation of datas')

def plot_clusters(clusters, k):
    color_norm = colors.Normalize(vmin=0, vmax=k-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hot')
    for i in range(k):
        plot_2d(clusters[i], scalar_map.to_rgba(i))
