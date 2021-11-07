"""
Evaluate using simple graph convolution networks.
"""
from shutil import which
import numpy as np
from matplotlib import pyplot as plt
import pdb

import scipy as sp
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg.eigen.arpack.arpack import eigs

from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

from munkres import Munkres
from tqdm.auto import tqdm


def convolution_step(features, uedge, vedge, num_conv=1):
    """
    """

    sx, sy = features.shape
    wedge = np.ones(len(uedge), dtype=np.float64)
    graph = sp.sparse.csr_matrix((wedge, (uedge, vedge)), shape=(sx, sx))
    graph = graph + graph.transpose()
    L = laplacian(graph, normed=True)
    adj_matrix = sp.sparse.eye(sx) - 0.5*L

    for _ in range(num_conv):
        features = adj_matrix.dot(features)
    return features


def convolution_step_weighted(features, uedge, vedge, wedge, num_conv=1, beta=1.0):
    """
    """

    sx, sy = features.shape
    wedge_sim = np.exp(-1*beta*wedge/(wedge.std()+1e-6))
    graph = sp.sparse.csr_matrix((wedge_sim, (uedge, vedge)), shape=(sx, sx))
    graph = graph + graph.transpose()

    L = laplacian(graph, normed=True)
    eigval, eigvec = eigsh(L, k=1, which='LM')
    adj_matrix = sp.sparse.eye(sx) - (1/np.max(eigval))*L
    for _ in range(num_conv):
        features = adj_matrix.dot(features)
    return features


def update_pred_labels_matching(pred_labels, gt_labels):
    """
    """
    indfilter = gt_labels != 0
    pred, gt = pred_labels[indfilter], gt_labels[indfilter]
    number_labels_pred = np.max(np.unique(pred_labels))
    number_labels_gt = len(np.unique(gt))
    C = confusion_matrix(gt, pred, labels=np.unique(np.sort(gt)))
    matching = Munkres()
    indexes = matching.compute((-1*(C.T)))
    map_arr = np.zeros(number_labels_pred+1, dtype=np.int64)
    for row, col in indexes:
        map_arr[row] = col+1

    return map_arr[pred_labels-1]


def cluster_OA_with_matching(pred_labels, gt_labels):
    """
    The number of classes should be the same.
    """
    pred_labels_match = update_pred_labels_matching(pred_labels, gt_labels)
    indfilter = gt_labels > 0
    return np.mean(pred_labels_match[indfilter] == gt_labels[indfilter])


def get_cluster_score(data, uedge, vedge, labels, wedge=None, beta=1.0):
    """
    """
    sx, sy = data.shape
    features = np.array(data.reshape((sx, sy)), copy=True)
    n_clusters = np.max(labels)
    max_score = 0.0
    score1 = []
    score2 = []

    if wedge is None:
        sx, sy = features.shape
        wedge = np.ones(len(uedge), dtype=np.float64)
        graph = sp.sparse.csr_matrix((wedge, (uedge, vedge)), shape=(sx, sx))
        graph = graph + graph.transpose()
        L = laplacian(graph, normed=True)
        adj_matrix = sp.sparse.eye(sx) - 0.5*L
    elif wedge is not None:
        sx, sy = features.shape
        # wedge_sim = np.exp(-1*beta*wedge/(wedge.std()+1e-6))+1e-6
        wedge_sim = 1-wedge
        graph = sp.sparse.csr_matrix((wedge_sim, (uedge, vedge)), shape=(sx, sx))
        graph = graph + graph.transpose()

        L = laplacian(graph, normed=True)
        eigval, eigvec = eigsh(L, k=1, which='LM')
        adj_matrix = sp.sparse.eye(sx) - (1/np.max(eigval))*L

    for _ in tqdm(range(200), desc='Spectral Eval', leave=False):
        features = adj_matrix.dot(features)
        score_tmp = []
        u, s, v = sp.sparse.linalg.svds(features, k=n_clusters, which='LM')
        for _ in range(1):
            kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(u)
            predict_labels = kmeans.predict(u)
            score_tmp.append(cluster_OA_with_matching(predict_labels, labels))

        max_score = max(max_score, np.mean(score_tmp))
        score1.append(np.mean(score_tmp))
        score2.append(max_score)

    return score1, score2


def evaluate_using_GCN(data, uedge, vedge, labels):
    """
    """
    tot = 10
    scores = []
    with tqdm(total=tot, desc='GCN', leave=False) as pbar:
        for i in range(tot):
            scores.append(get_cluster_score(data, uedge, vedge, labels))
            pbar.update()
            pbar.set_postfix({'mean': '{:0.2f}'.format(np.mean(scores)), 'var': '{:0.4f}'.format(np.std(scores))})

    return np.mean(scores), np.std(scores)


def evaluate_using_GCN_weighted(data, uedge, vedge, wedge, labels):
    """
    """
    tot = 10
    scores = []
    with tqdm(total=tot, desc='GCN', leave=False) as pbar:
        for i in range(tot):
            scores.append(get_cluster_score(data, uedge, vedge, labels, wedge))
            pbar.update()
            pbar.set_postfix({'mean': '{:0.2f}'.format(np.mean(scores)), 'var': '{:0.4f}'.format(np.std(scores))})

    return np.mean(scores), np.std(scores)
