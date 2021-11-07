import numpy as np
import os
import pdb
from matplotlib import pyplot as plt

from Datasets import get_indianpines_data, get_paviaCentre_data, get_botswana_data, get_ksc_data, get_salinas_data
from Utils import get_4adj_edges, estimate_weights, visualize_4adj_edge_weights
from CacheEdgeWeightEstimate import get_edge_weight_estimate
from Evaluate_RandomWalk import evaluate_using_rw, evaluate_using_rw_weighted


def check_results(fname, dataset, method, num_samples):
    """Check if we already computed the results!!
    """
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith("{},{},{}".format(dataset, method, num_samples)):
                return True
    return False


# Create the file if it does not exist.
if not os.path.exists("./dump/Results_ScoreRW.txt"):
    with open("./dump/Results_ScoreRW.txt", 'w') as f:
        f.write("Dataset,Method,NumSamples,OA_mean,OA_stdev,AA_mean,AA_stdev,Kappa_mean,Kappa_stdev\n")
else:
    flag = False
    with open("./dump/Results_ScoreRW.txt", 'r') as f:
        for line in f:
            if line.startswith("Dataset,Method,NumSamples,OA_mean,OA_stdev,AA_mean,AA_stdev,Kappa_mean,Kappa_stdev"):
                flag = True
    if not flag:
        with open("./dump/Results_ScoreRW.txt", 'w') as f:
            f.write("Dataset,Method,NumSamples,OA_mean,OA_stdev,AA_mean,AA_stdev,Kappa_mean,Kappa_stdev\n")

for dataset in ['Indianpines', 'Botswana', 'KSC', 'PaviaCentre', 'Salinas']:

    if dataset == 'Indianpines':
        data, labels = get_indianpines_data()
    elif dataset == 'PaviaCentre':
        data, labels = get_paviaCentre_data()
    elif dataset == 'Salinas':
        data, labels = get_salinas_data()
    elif dataset == 'Botswana':
        data, labels = get_botswana_data()
    elif dataset == 'KSC':
        data, labels = get_ksc_data()

    sx, sy, sz = data.shape
    data_flatten = data.reshape((sx*sy, sz))
    uedge, vedge = get_4adj_edges(data)

    for num_samples in [5, 10, 15, 20, 25]:
        if check_results("./dump/Results_ScoreRW.txt", dataset, 'Unweighted', num_samples):
            continue
        res_unweighted = evaluate_using_rw(data_flatten, uedge, vedge, labels.flatten(), num_samples)
        with open("./dump/Results_ScoreRW.txt", 'a') as f:
            f.write("{},{},{},{},{},{},{},{},{}\n".format(dataset, 'Unweighted', num_samples, *tuple(res_unweighted)))

    for num_samples in [5, 10, 15, 20, 25]:
        if check_results("./dump/Results_ScoreRW.txt", dataset, 'Cosine', num_samples):
            continue
        res_cosine_similarity = evaluate_using_rw_weighted(data_flatten, uedge, vedge, labels.flatten(), num_samples=num_samples, sim='cosine')
        with open("./dump/Results_ScoreRW.txt", 'a') as f:
            f.write("{},{},{},{},{},{},{},{},{}\n".format(dataset, 'Cosine', num_samples, *tuple(res_cosine_similarity)))

    wedge = get_edge_weight_estimate("./cache/", dataset)
    for num_samples in [5, 10, 15, 20, 25]:
        if check_results("./dump/Results_ScoreRW.txt", dataset, 'EnsembleEdgeWeights', num_samples):
            continue
        res_ensemble_edgeweights = evaluate_using_rw_weighted(data_flatten, uedge, vedge, labels.flatten(), num_samples=num_samples, wedge=wedge)
        with open("./dump/Results_ScoreRW.txt", 'a') as f:
            f.write("{},{},{},{},{},{},{},{},{}\n".format(dataset, 'EnsembleEdgeWeights', num_samples, *tuple(res_ensemble_edgeweights)))
