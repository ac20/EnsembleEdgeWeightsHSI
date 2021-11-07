import numpy as np
import pdb
from matplotlib import pyplot as plt
from skimage import exposure

from Datasets import get_indianpines_data, get_paviaCentre_data, get_botswana_data, get_ksc_data, get_salinas_data
from Utils import get_4adj_edges, estimate_weights, visualize_4adj_edge_weights
from CacheEdgeWeightEstimate import get_edge_weight_estimate


for dataset in ['Indianpines', 'PaviaCentre', 'Salinas', 'Botswana', 'KSC']:

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

    uedge, vedge = get_4adj_edges(data)

    wedge = get_edge_weight_estimate(root="./cache/", dataset=dataset)
    img = visualize_4adj_edge_weights(data, uedge, vedge, wedge)
    plt.figure()
    plt.imshow(1-img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./dump/Experiment1_" + dataset + "_boundary.png",  bbox_inches='tight')

    plt.figure()
    plt.imshow(labels)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./dump/Experiment1_" + dataset + "_gt.png",  bbox_inches='tight')

    sx, sy, sz = data.shape
    data_flatten = data.reshape((sx*sy, sz))
    wedge = np.sqrt(np.sum((data_flatten[uedge, :]-data_flatten[vedge, :])**2, axis=-1))
    img = visualize_4adj_edge_weights(data, uedge, vedge, wedge)
    plt.figure()
    plt.imshow(1-img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./dump/Experiment1_" + dataset + "_boundarySimple.png",  bbox_inches='tight')
