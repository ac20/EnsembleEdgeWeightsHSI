import numpy as np
import os
import pdb
from matplotlib import pyplot as plt

from Datasets import get_indianpines_data, get_paviaCentre_data, get_botswana_data, get_ksc_data, get_paviaCentre_data_trimmed, get_salinas_data
from Utils import get_4adj_edges, estimate_weights, visualize_4adj_edge_weights
from CacheEdgeWeightEstimate import get_edge_weight_estimate
from Evaluate_GCN import get_cluster_score


def check_results(fname, dataset, method, rep):
    """Check if we already computed the results!!
    """
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith("{},{},{}".format(dataset, method, rep)):
                return True
    return False


# Create the file if it does not exist.
if not os.path.exists("./dump/Results_ScoreGCNV3.txt"):
    with open("./dump/Results_ScoreGCNV3.txt", 'w') as f:
        f.write("Dataset,Method,rep,iIter,scoreIter,scoreMax\n")
else:
    flag = False
    with open("./dump/Results_ScoreGCNV3.txt", 'r') as f:
        for line in f:
            if line.startswith("Dataset,Method,rep,iIter,scoreIter,scoreMax"):
                flag = True
    if not flag:
        with open("./dump/Results_ScoreGCNV3.txt", 'w') as f:
            f.write("Dataset,Method,rep,iIter,scoreIter,scoreMax\n")

for dataset in ['Salinas', 'Indianpines', 'PaviaCentre', 'PaviaCentre_trimmed']:

    if dataset == 'Indianpines':
        data, labels = get_indianpines_data()
    elif dataset == 'PaviaCentre':
        data, labels = get_paviaCentre_data()
    elif dataset == 'PaviaCentre_trimmed':
        data, labels = get_paviaCentre_data_trimmed()
    elif dataset == 'Salinas':
        data, labels = get_salinas_data()
    elif dataset == 'Botswana':
        data, labels = get_botswana_data()
    elif dataset == 'KSC':
        data, labels = get_ksc_data()

    sx, sy, sz = data.shape
    data_flatten = data.reshape((sx*sy, sz))
    uedge, vedge = get_4adj_edges(data)
    labels_flatten = labels.flatten()

    for rep in range(10):
        if not check_results("./dump/Results_ScoreGCNV3.txt", dataset, 'Unweighted', rep):
            score_all, score_max = get_cluster_score(data_flatten, uedge, vedge, labels_flatten)
            for iIter in range(len(score_all)):
                with open("./dump/Results_ScoreGCNV3.txt", 'a') as f:
                    f.write("{},{},{},{},{},{}\n".format(dataset, 'Unweighted', rep, iIter, score_all[iIter], score_max[iIter]))

        wedge = get_edge_weight_estimate("./cache/", dataset)
        if not check_results("./dump/Results_ScoreGCNV3.txt", dataset, 'Weighted', rep):
            score_all, score_max = get_cluster_score(data_flatten, uedge, vedge, labels_flatten, wedge)
            for iIter in range(len(score_all)):
                with open("./dump/Results_ScoreGCNV3.txt", 'a') as f:
                    f.write("{},{},{},{},{},{}\n".format(dataset, 'Weighted', rep, iIter, score_all[iIter], score_max[iIter]))
