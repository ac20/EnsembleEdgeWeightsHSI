"""
Read the datasets.
"""

import numpy as np
import pdb
import os
import wget

import scipy as sp
from scipy.io import loadmat

from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs


def _process_data_pca(X, y):
    sx, sy, sz = np.shape(X)
    X = X.reshape((-1, sz))
    X = PCA(n_components=sz).fit_transform(X)
    X = X.reshape((sx, sy, sz))
    muX = np.mean(X, axis=(0, 1), keepdims=True)
    stdX = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    X = (X - muX)/stdX
    return X, y


def _process_data_mean(X, y):
    muX = np.mean(X, axis=(0, 1), keepdims=True)
    stdX = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    X = (X - muX)/stdX
    return X, y


def get_indianpines_data(root="./data", download=True, preprocess='mean'):
    """
    """
    data_url = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
    gt_url = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"

    if (not os.path.exists(root+"/Indian_pines_corrected.mat")) and download:
        wget.download(data_url, root + "/Indian_pines_corrected.mat")

    if (not os.path.exists(root+"/Indian_pines_gt.mat")) and download:
        wget.download(gt_url, root + "/Indian_pines_gt.mat")

    data = loadmat("./data/Indian_pines_corrected.mat")
    X = np.array(data['indian_pines_corrected'], dtype=np.float32)

    data = loadmat("./data/Indian_pines_gt.mat")
    y = np.array(data['indian_pines_gt'], dtype=np.int32)

    if preprocess == 'mean':
        X, y = _process_data_mean(X, y)
    elif preprocess == 'pca':
        X, y = _process_data_pca(X, y)
    return X, y


def get_paviaU_data(root="./data", download=True, preprocess='mean'):
    """
    """
    data_url = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
    gt_url = "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat"

    if (not os.path.exists(root+"/PaviaU.mat")) and download:
        wget.download(data_url, root + "/PaviaU.mat")

    if (not os.path.exists(root+"/PaviaU_gt.mat")) and download:
        wget.download(gt_url, root + "/PaviaU_gt.mat")

    data = loadmat("./data/PaviaU.mat")
    X = np.array(data['paviaU'], dtype=np.float32)

    # # Normalize
    # muX = np.mean(X, axis=(0, 1), keepdims=True)
    # stdX = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    # X = (X - muX)/stdX

    data = loadmat("./data/PaviaU_gt.mat")
    y = np.array(data['paviaU_gt'], dtype=np.int32)

    if preprocess == 'mean':
        X, y = _process_data_mean(X, y)
    elif preprocess == 'pca':
        X, y = _process_data_pca(X, y)
    return X, y


def get_salinas_data(root="./data", download=True, preprocess='mean'):
    """
    """
    data_url = "http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat"
    gt_url = "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat"

    if (not os.path.exists(root+"/Salinas_corrected.mat")) and download:
        wget.download(data_url, root + "/Salinas_corrected.mat")

    if (not os.path.exists(root+"/Salinas_gt.mat")) and download:
        wget.download(gt_url, root + "/Salinas_gt.mat")

    data = loadmat("./data/Salinas_corrected.mat")
    X = np.array(data['salinas_corrected'], dtype=np.float32)

    # # Normalize
    # muX = np.mean(X, axis=(0, 1), keepdims=True)
    # stdX = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    # X = (X - muX)/stdX

    data = loadmat("./data/Salinas_gt.mat")
    y = np.array(data['salinas_gt'], dtype=np.int32)

    if preprocess == 'mean':
        X, y = _process_data_mean(X, y)
    elif preprocess == 'pca':
        X, y = _process_data_pca(X, y)
    return X, y


def get_paviaCentre_data(root="./data", download=True, preprocess='mean'):
    """
    """
    data_url = "http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat"
    gt_url = "http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat"

    if (not os.path.exists(root+"/Pavia.mat")) and download:
        wget.download(data_url, root + "/Pavia.mat")

    if (not os.path.exists(root+"/Pavia_gt.mat")) and download:
        wget.download(gt_url, root + "/Pavia_gt.mat")

    data = loadmat("./data/Pavia.mat")
    X = np.array(data['pavia'], dtype=np.float32)

    # # Normalize
    # muX = np.mean(X, axis=(0, 1), keepdims=True)
    # stdX = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    # X = (X - muX)/stdX

    data = loadmat("./data/Pavia_gt.mat")
    y = np.array(data['pavia_gt'], dtype=np.int32)

    if preprocess == 'mean':
        X, y = _process_data_mean(X, y)
    elif preprocess == 'pca':
        X, y = _process_data_pca(X, y)
    return X, y


def get_paviaCentre_data_trimmed(root="./data", download=True, preprocess='mean'):
    """
    """
    data_url = "http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat"
    gt_url = "http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat"

    if (not os.path.exists(root+"/Pavia.mat")) and download:
        wget.download(data_url, root + "/Pavia.mat")

    if (not os.path.exists(root+"/Pavia_gt.mat")) and download:
        wget.download(gt_url, root + "/Pavia_gt.mat")

    data = loadmat("./data/Pavia.mat")
    X = np.array(data['pavia'], dtype=np.float32)

    # # Normalize
    # muX = np.mean(X, axis=(0, 1), keepdims=True)
    # stdX = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    # X = (X - muX)/stdX

    data = loadmat("./data/Pavia_gt.mat")
    y = np.array(data['pavia_gt'], dtype=np.int32)

    if preprocess == 'mean':
        X, y = _process_data_mean(X, y)
    elif preprocess == 'pca':
        X, y = _process_data_pca(X, y)
    return X[:, -490:, :], y[:, -490:]


def get_ksc_data(root="./data", download=True, preprocess='mean'):
    """
    """
    data_url = "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat"
    gt_url = "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat"

    if (not os.path.exists(root+"/KSC.mat")) and download:
        wget.download(data_url, root + "/KSC.mat")

    if (not os.path.exists(root+"/KSC_gt.mat")) and download:
        wget.download(gt_url, root + "/KSC_gt.mat")

    data = loadmat("./data/KSC.mat")
    X = np.array(data['KSC'], dtype=np.float32)

    # # Normalize
    # muX = np.mean(X, axis=(0, 1), keepdims=True)
    # stdX = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    # X = (X - muX)/stdX

    data = loadmat("./data/KSC_gt.mat")
    y = np.array(data['KSC_gt'], dtype=np.int32)

    if preprocess == 'mean':
        X, y = _process_data_mean(X, y)
    elif preprocess == 'pca':
        X, y = _process_data_pca(X, y)
    return X, y


def get_houston_data(root="./data", download=False, preprocess='mean'):
    """ 
    ** Links for houston dataset are not working as on April 27, 2021.
    Please download the files manually from
    http://hyperspectral.ee.uh.edu/2egf4tg8hial13gt/2013_DFTC.zip
    and unzip them.
    """

    data = loadmat("./data/houston.mat")
    X = np.array(data['houston'], dtype=np.float32)

    # Normalize
    # muX = np.mean(X, axis=(0, 1), keepdims=True)
    # stdX = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    # X = (X - muX)/stdX

    data = loadmat("./data/houston_gt.mat")
    y1 = np.array(data['houston_gt_tr'], dtype=np.int32)
    y2 = np.array(data['houston_gt_te'], dtype=np.int32)
    y2[y1 > 0] = y1[y1 > 0]
    y = np.array(y2, dtype=np.int32)

    if preprocess == 'mean':
        X, y = _process_data_mean(X, y)
    elif preprocess == 'pca':
        X, y = _process_data_pca(X, y)
    return X, y


def get_gaussian_blobs_hsi_data():
    """
    """
    mu1 = np.concatenate((np.ones(40, dtype=np.float32), np.ones(40, dtype=np.float32), np.zeros(40, dtype=np.float32)))
    mu2 = np.concatenate((np.zeros(40, dtype=np.float32), np.ones(40, dtype=np.float32), np.ones(40, dtype=np.float32)))
    mu3 = np.concatenate((np.ones(40, dtype=np.float32), np.zeros(40, dtype=np.float32), np.zeros(40, dtype=np.float32)))

    X, y = make_blobs(n_samples=50*50, n_features=120, centers=[mu1, mu2, mu3], cluster_std=1.5, random_state=77)
    X = X.reshape((50, 50, 120))
    X, y = _process_data_mean(X, y)
    return X, y


def get_botswana_data(root="./data", download=True, preprocess='mean'):
    """
    """
    data_url = "http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat"
    gt_url = "http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat"

    if (not os.path.exists(root+"/Botswana.mat")) and download:
        wget.download(data_url, root + "/Botswana.mat")

    if (not os.path.exists(root+"/Botswana_gt.mat")) and download:
        wget.download(gt_url, root + "/Botswana_gt.mat")

    data = loadmat("./data/Botswana.mat")
    X = np.array(data['Botswana'], dtype=np.float32)

    data = loadmat("./data/Botswana_gt.mat")
    y = np.array(data['Botswana_gt'], dtype=np.int32)

    if preprocess == 'mean':
        X, y = _process_data_mean(X, y)
    elif preprocess == 'pca':
        X, y = _process_data_pca(X, y)
    return X, y
