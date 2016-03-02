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

import matplotlib.cm as cmx
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.patches import Ellipse

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

def plot_3d(data, title='', alpha=0.2):
    """Scatter plot of 3D data."""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    maxi = max([max(abs(el)) for el in data])
    ax.set_xlim([-maxi, maxi])
    ax.set_ylim([-maxi, maxi])
    ax.set_zlim([-maxi, maxi])
    data_t = data.T
    ax.scatter(data_t[0], data_t[1], data_t[2], alpha=alpha)
    plt.title(title)


def plot_2d(datas, color='r', title='', maxi=None, fig=None, alpha=0.2):
    """Scatter plot of 2D data."""
    datas_t = np.array(datas)
    if maxi:
        ax = fig.gca()
        ax.set_xlim([-maxi, maxi])
        ax.set_ylim([-maxi, maxi])
    plt.scatter(datas_t[:,0], datas_t[:,1], alpha=alpha, c=color)
    plt.title(title)


def plot_clusters(clusters, k):
    """Plot each of the given cluster in a different color. clusters is a
    list of list."""
    # generate a colormap of size k
    color_norm = colors.Normalize(vmin=0, vmax=k-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hot')
    # plot each cluster with a given color of the generated colormap
    for i, cluster in range(k):
        plot_2d(clusters[i], scalar_map.to_rgba(i))


def plot_multivariate_ellipse(multivariates, K):
    """Plot confidence ellipses of the given multivariates."""
    # Generate a colormap to attribute a color to each cluster
    color_norm = colors.Normalize(vmin=0, vmax=K-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='autumn')

    for i, m in enumerate(multivariates):
        vals, vecs = np.linalg.eigh(m.cov)
        # order = vals.argsort()[::-1]
        # vals, vecs = vals[order], vecs[:,order]
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        width, height = 4 * np.sqrt(vals)
        plt.scatter(m.mean[0], m.mean[1], c=scalar_map.to_rgba(i))
        ellip = Ellipse(xy=m.mean, width=width, height=height,
                        angle=theta, fill=False, color=scalar_map.to_rgba(i))
        plt.gca().add_artist(ellip)


class Arrow3D(FancyArrowPatch):
    """Arrows in 3D. The code was found on stackoverflow but I cannot 
    remember on which question exactly."""
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]), (xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
