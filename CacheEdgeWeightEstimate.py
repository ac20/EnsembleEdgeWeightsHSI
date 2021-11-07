import numpy as np
import pdb
from matplotlib import pyplot as plt

from Datasets import get_indianpines_data, get_paviaCentre_data, get_botswana_data, get_ksc_data, get_salinas_data, get_paviaCentre_data_trimmed
from Utils import get_4adj_edges, estimate_weights


def get_edge_weight_estimate(root="./cache/", dataset='indianpines'):
    """
    """
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

    fname = root + "/edge_weight_" + dataset + ".npy"
    try:
        wedge = np.load(fname)
        print("Reading the data from cache...")
    except:
        print("Unable to load edge weights for {} data...".format(dataset))
        sx, sy, sz = data.shape
        data_flatten = data.reshape((sx*sy, sz))
        uedge, vedge = get_4adj_edges(data)
        wedge = estimate_weights(uedge, vedge, data_flatten)
        np.save(fname, wedge, allow_pickle=False)
    return wedge


if __name__ == "__main__":
    get_edge_weight_estimate(root="./cache", dataset='Indianpines')
    get_edge_weight_estimate(root="./cache", dataset='PaviaCentre')
    get_edge_weight_estimate(root="./cache", dataset='Salinas')
    get_edge_weight_estimate(root="./cache", dataset='Botswana')
    get_edge_weight_estimate(root="./cache", dataset='KSC')
