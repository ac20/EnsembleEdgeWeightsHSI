"""
"""

import numpy as np
import pdb

import scipy as sp

import higra as hg
from tqdm.auto import tqdm

from Evaluate_RandomWalk import random_walk, random_walk_matlab


def get_4adj_edges(img):
    """
    """
    s0, s1 = img.shape[:2]
    z = np.arange(s0*s1).reshape((s0, s1))
    uedge = np.concatenate((z[:-1, :].flatten(), z[:, :-1].flatten()))
    vedge = np.concatenate((z[1:, :].flatten(), z[:, 1:].flatten()))
    assert len(uedge) == len(vedge)
    assert len(uedge) == (s0-1)*s1 + (s1-1)*s0
    return uedge, vedge


def _generate_sample_points_random(size_data):
    """
    Note: we consider sqrt(size_data) to be the number of random nodes considered.
    """
    while True:
        arr = np.random.choice(np.arange(size_data), size=int(np.sqrt(size_data)), replace=False)
        yield np.array(arr, dtype=np.int64)


def estimate_weights(uedge, vedge, data):
    """
    """
    size_data, num_features = np.shape(data)
    sample_generator = _generate_sample_points_random(size_data)

    graph = hg.UndirectedGraph()
    graph.add_vertices(size_data)
    graph.add_edges(uedge, vedge)
    wedge_estimate = np.zeros(uedge.shape)

    # Estimate base wedge
    rep = 100
    for _ in tqdm(range(rep), leave=True, desc="Estimating Edge-Weights..."):
        sample_points = next(sample_generator)
        for _ in range(rep):
            try:
                subset_features = np.random.choice(np.arange(num_features), size=16, replace=False)
            except:
                subset_features = np.random.choice(np.arange(num_features), size=16, replace=True)
            data_flat_subset = data[:, subset_features]
            wedge = np.sqrt(np.sum((data_flat_subset[uedge, :] - data_flat_subset[vedge, :])**2, axis=-1))

            seeds = np.zeros(size_data, dtype=np.int64)
            seeds[sample_points] = np.arange(len(sample_points)) + 1
            tmp = hg.labelisation_seeded_watershed(graph, wedge, seeds)
            wedge_estimate += (tmp[uedge] != tmp[vedge])*1

    wedge_estimate = wedge_estimate/(rep**2)
    return wedge_estimate


def estimate_weights_RW(uedge, vedge, data):
    """
    """
    size_data, num_features = np.shape(data)
    sample_generator = _generate_sample_points_random(size_data)

    wedge_estimate = np.zeros(uedge.shape)

    # Estimate base wedge
    rep = 100
    for _ in tqdm(range(rep), leave=True, desc="Estimating Edge-Weights..."):
        sample_points = next(sample_generator)
        for _ in range(rep):
            try:
                subset_features = np.random.choice(np.arange(num_features), size=16, replace=False)
            except:
                subset_features = np.random.choice(np.arange(num_features), size=16, replace=True)
            data_flat_subset = data[:, subset_features]

            seeds = np.zeros(size_data, dtype=np.int64)
            seeds[sample_points] = np.arange(len(sample_points)) + 1
            tmp = random_walk_matlab(data_flat_subset, seeds, uedge, vedge)
            wedge_estimate += (tmp[uedge] != tmp[vedge])*1

    wedge_estimate = wedge_estimate/(rep**2)
    return wedge_estimate


def visualize_4adj_edge_weights(data, uedge, vedge, wedge):
    """Returns an image which visualizes the edge weights

    Note : It is known that higra orders the edges uedge<vedge
    """

    assert len(data.shape) == 3, "Data should have dimensions - (nH, nW, nBands)"
    sx, sy, sz = data.shape

    grid_graph = hg.get_4_adjacency_graph((sx, sy))
    uedge_4adj, vedge_4adj = grid_graph.edge_list()

    maxval = sx*sy
    arg_val = np.argsort(maxval*uedge + vedge)

    wedge_4adj = np.zeros(len(uedge_4adj))
    arg_val_4adj = np.argsort(maxval*uedge_4adj + vedge_4adj)

    wedge_4adj[arg_val_4adj] = wedge[arg_val]

    img = hg.graph_4_adjacency_2_khalimsky(grid_graph, wedge_4adj, add_extra_border=True)
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    return img
