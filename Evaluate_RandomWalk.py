"""
Evaluate a set of edges using random walk.
"""

import numpy as np
from numpy.core.fromnumeric import size

import warnings
import pdb
from tqdm.auto import tqdm
import matlab.engine

import scipy as sp
from scipy.sparse.linalg import cg, spsolve

from sklearn.metrics import cohen_kappa_score

try:
    from scipy.sparse.linalg.dsolve import umfpack
    old_del = umfpack.UmfpackContext.__del__

    def new_del(self):
        try:
            old_del(self)
        except AttributeError:
            pass
    umfpack.UmfpackContext.__del__ = new_del
    UmfpackContext = umfpack.UmfpackContext()
except ImportError:
    UmfpackContext = None

try:
    from pyamg import ruge_stuben_solver
    amg_loaded = True
except ImportError:
    amg_loaded = False


def _build_linear_system(data, uedge, vedge, seeds, sim='euclid', beta=1.0, weights=None):
    """
    """
    nlabels = np.max(seeds)
    indices = np.arange(seeds.size)
    seeds_mask = seeds > 0
    unlabeled_indices = indices[~seeds_mask]
    seeds_indices = indices[seeds_mask]

    if sim == 'cosine':
        X1, X2 = data[uedge, :], data[vedge, :]
        normX1 = np.sqrt(np.sum(X1**2, axis=-1))
        normX2 = np.sqrt(np.sum(X2**2, axis=-1))
        wedge = np.sum(X1*X2, axis=-1)/(normX1*normX2) + 1e-6
    elif sim == 'euclid':
        X1, X2 = data[uedge, :], data[vedge, :]
        dist = np.sqrt(np.sum((X1-X2)**2, axis=-1)) + 1e-6
        wedge = np.exp(-1*beta*dist/dist.std()) + 1e-6
    elif sim == 'manual':
        assert weights is not None, "Weights argument should be given for sim='manual'"
        wedge = np.exp(-1*beta*weights/weights.std()) + 1e-6

    graph = sp.sparse.csr_matrix((wedge, (uedge, vedge)), shape=(seeds.size, seeds.size))
    graph = graph + graph.transpose()

    # Assert graph is connected
    n, _ = sp.sparse.csgraph.connected_components(graph, directed=False)
    assert n == 1, "Graph is not connected!!!"

    lap_sparse = sp.sparse.csgraph.laplacian(graph, normed=False).tocsr()

    rows = lap_sparse[unlabeled_indices, :]
    lap_sparse = rows[:, unlabeled_indices]
    B = -rows[:, seeds_indices]

    seeds = seeds[seeds_mask]
    seeds_mask = sp.sparse.csc_matrix(np.hstack(
        [np.atleast_2d(seeds == lab).T for lab in range(1, nlabels + 1)]))
    rhs = B.dot(seeds_mask)

    return lap_sparse, rhs


def _solve_linear_system(lap_sparse, B, tol, mode):

    if mode is None:
        mode = 'cg_j'

    if mode == 'cg_mg' and not amg_loaded:
        warnings.warn('"cg_mg" not available, it requires pyamg to be installed. '
                      'The "cg_j" mode will be used instead.',
                      stacklevel=2)
        mode = 'cg_j'

    if mode == 'bf':
        X = spsolve(lap_sparse, B.toarray()).T
    else:
        maxiter = None
        if mode == 'cg':
            if UmfpackContext is None:
                warnings.warn('"cg" mode may be slow because UMFPACK is not available. '
                              'Consider building Scipy with UMFPACK or use a '
                              'preconditioned version of CG ("cg_j" or "cg_mg" modes).',
                              stacklevel=2)
            M = None
        elif mode == 'cg_j':
            M = sp.sparse.diags(1.0 / lap_sparse.diagonal())
        else:
            # mode == 'cg_mg'
            lap_sparse = lap_sparse.tocsr()
            ml = ruge_stuben_solver(lap_sparse)
            M = ml.aspreconditioner(cycle='V')
            maxiter = 30
        cg_out = [
            cg(lap_sparse, B[:, i].toarray(), tol=tol, M=M, maxiter=maxiter)
            for i in range(B.shape[1])]
        if np.any([info > 0 for _, info in cg_out]):
            warnings.warn("Conjugate gradient convergence to tolerance not achieved. "
                          "Consider decreasing beta to improve system conditionning.",
                          stacklevel=2)
        X = np.asarray([x for x, _ in cg_out])

    return X


def random_walk(data, seeds, uedge, vedge):
    """

    Notes:
    1. Unlabelled nodes are denoted by {seeds == 0}.
    2. For {seeds > 0}, labels are given as integers 1, 2, ..., k. 
    """
    lap_sparse, B = _build_linear_system(data, uedge, vedge, seeds)
    X = _solve_linear_system(lap_sparse, B, tol=1e-3, mode='cg_j')

    rw_labels = np.array(seeds, copy=True)
    mask = np.where(seeds == 0)
    rw_labels[mask] = np.argmax(X, axis=0) + 1
    return rw_labels


def random_walk_matlab(data, seeds, uedge, vedge):
    """
    """
    lap_sparse, B = _build_linear_system(data, uedge, vedge, seeds)

    tmp = lap_sparse.tocoo()
    i, j, s = matlab.int64(list(tmp.row+1)), matlab.int64(list(tmp.col+1)), matlab.double(list(tmp.data))
    m, n = tmp.shape
    tmp2 = B.tocoo()
    Bi, Bj, Bs = matlab.int64(list(tmp2.row+1)), matlab.int64(list(tmp2.col+1)), matlab.double(list(tmp2.data))
    Bm, Bn = tmp2.shape
    eng = matlab.engine.start_matlab()
    X = eng.solve(i, j, s, m, n, Bi, Bj, Bs, Bm, Bn)
    X = np.array(X)
    eng.quit()

    rw_labels = np.array(seeds, copy=True)
    mask = np.where(seeds == 0)
    rw_labels[mask] = np.argmax(X, axis=1) + 1
    return rw_labels


def random_walk_with_weight(data, seeds, uedge, vedge, wedge=None, sim=None):
    """

    Notes:
    1. Unlabelled nodes are denoted by {seeds == 0}.
    2. For {seeds > 0}, labels are given as integers 1, 2, ..., k. 
    """
    if sim == 'cosine':
        lap_sparse, B = _build_linear_system(data, uedge, vedge, seeds, sim='cosine')
    elif (sim is None) and wedge is not None:
        lap_sparse, B = _build_linear_system(data, uedge, vedge, seeds, sim='manual', beta=1.0, weights=wedge)
    else:
        raise Exception("either sim='cosine' or wege is not None!")
    X = _solve_linear_system(lap_sparse, B, tol=1e-3, mode='bf')

    rw_labels = np.array(seeds, copy=True)
    mask = np.where(seeds == 0)
    rw_labels[mask] = np.argmax(X, axis=0) + 1
    return rw_labels


def _random_seeds(labels_arr, num_samples=5):
    seeds = np.zeros(len(labels_arr), dtype=np.int64)
    for label in range(1, np.max(labels_arr)+1):
        tmp = np.where(labels_arr == label)[0]
        try:
            seeds_choose = np.random.choice(tmp, size=num_samples, replace=False)
        except:
            seeds_choose = np.random.choice(tmp, size=num_samples, replace=True)
        seeds[seeds_choose] = label
    return seeds


def _compute_OA(predlabels, labels, seeds):
    """
    """
    indtrain, indtest = np.where(seeds > 0)[0], np.where(seeds == 0)[0]
    acc_train = np.mean(predlabels[indtrain] == labels[indtrain])
    acc_test = np.mean(predlabels[indtest] == labels[indtest])
    return acc_train, acc_test


def _compute_AA(predlabels, labels, seeds):
    """
    """
    indtrain, indtest = np.where(seeds > 0)[0], np.where(seeds == 0)[0]
    acc_train = 0
    acc_test = 0
    count_train, count_test = 0, 0
    for l in range(np.max(labels)):
        if np.sum(labels[indtrain] == l+1) > 0:
            indselect = indtrain[labels[indtrain] == l+1]
            acc_train += np.mean(predlabels[indselect] == labels[indselect])
            count_train += 1
        if np.sum(labels[indtest] == l+1) > 0:
            indselect = indtest[labels[indtest] == l+1]
            acc_test += np.mean(predlabels[indselect] == labels[indselect])
            count_test += 1
    return acc_train/count_train, acc_test/count_test


def _compute_accuracy_classwise(predlabels, labels, seeds):
    """
    """
    indtrain, indtest = np.where(seeds > 0)[0], np.where(seeds == 0)[0]
    acc_train = []
    acc_test = []
    count = 0
    for l in range(np.max(labels)):
        indselect = indtrain[labels[indtrain] == l+1]
        acc_train.append(np.mean(predlabels[indselect] == labels[indselect]))
        indselect = indtest[labels[indtest] == l+1]
        acc_test.append(np.mean(predlabels[indselect] == labels[indselect]))
        count += 1
    return acc_train, acc_test


def _compute_kappa(predlabels, labels, seeds):
    """
    """
    indtrain, indtest = np.where(seeds > 0)[0], np.where(seeds == 0)[0]
    kappa_train = cohen_kappa_score(predlabels[indtrain], labels[indtrain])
    kappa_test = cohen_kappa_score(predlabels[indtest], labels[indtest])
    return kappa_train, kappa_test


def evaluate_using_rw(data, uedge, vedge, labels, num_samples=5):
    """
    Note that {labels==0} is the set of pixels which do not have ground_truth. We consider these
    as unlabelled as well. However, these pixels are ignored for evaluation.
    """

    list_OA = []
    list_AA = []
    list_Kappa = []
    for rep in tqdm(range(50), desc='Evaluate Random Walk', leave=False):
        seeds = _random_seeds(labels, num_samples)
        rw_labels = random_walk(data, seeds, uedge, vedge)
        mask = labels > 0
        rw_labels, labels_mask, seeds_mask = rw_labels[mask], labels[mask], seeds[mask]
        OA_train, OA_test = _compute_OA(rw_labels, labels_mask, seeds_mask)
        AA_train, AA_test = _compute_AA(rw_labels, labels_mask, seeds_mask)
        Kappa_train, Kappa_test = _compute_kappa(rw_labels, labels_mask, seeds_mask)
        list_OA.append(OA_test)
        list_AA.append(AA_test)
        list_Kappa.append(Kappa_test)

    return np.mean(list_OA), np.std(list_OA), np.mean(list_AA), np.std(list_AA), np.mean(list_Kappa), np.std(list_Kappa)


def evaluate_using_rw_weighted(data, uedge, vedge, labels, num_samples=5, wedge=None,  sim=None):
    """
    Note that {labels==0} is the set of pixels which do not have ground_truth. We consider these
    as unlabelled as well. However, these pixels are ignored for evaluation.
    """

    list_OA = []
    list_AA = []
    list_Kappa = []
    for rep in tqdm(range(50), desc='Evaluate Random Walk', leave=False):
        seeds = _random_seeds(labels, num_samples)
        rw_labels = random_walk_with_weight(data, seeds, uedge, vedge, wedge=wedge, sim=sim)
        mask = labels > 0
        rw_labels, labels_mask, seeds_mask = rw_labels[mask], labels[mask], seeds[mask]
        OA_train, OA_test = _compute_OA(rw_labels, labels_mask, seeds_mask)
        AA_train, AA_test = _compute_AA(rw_labels, labels_mask, seeds_mask)
        Kappa_train, Kappa_test = _compute_kappa(rw_labels, labels_mask, seeds_mask)
        list_OA.append(OA_test)
        list_AA.append(AA_test)
        list_Kappa.append(Kappa_test)

    return np.mean(list_OA), np.std(list_OA), np.mean(list_AA), np.std(list_AA), np.mean(list_Kappa), np.std(list_Kappa)
