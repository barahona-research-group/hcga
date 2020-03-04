"""make synthetic datasets: WIP!!"""
import os
import re
import shutil
import zipfile

import networkx as nx
import numpy as np
import wget

from ..io import save_dataset



def synthetic_data():

    graphs = [
        nx.planted_partition_graph(
            1, 10, rd.uniform(0.6, 1), rd.uniform(0.1, 0.4), seed=None, directed=False
        )
        for i in range(50)
    ] + [
        nx.planted_partition_graph(
            2, 5, rd.uniform(0.6, 1), rd.uniform(0.1, 0.4), seed=None, directed=False
        )
        for i in range(50)
    ]

    for i in range(len(graphs)):
        if not nx.is_connected(graphs[i]):
            if i < 50:
                graphs[i] = nx.planted_partition_graph(
                    1,
                    10,
                    rd.uniform(0.6, 1),
                    rd.uniform(0.1, 0.4),
                    seed=None,
                    directed=False,
                )
            elif i > 50:
                graphs[i] = nx.planted_partition_graph(
                    2,
                    5,
                    rd.uniform(0.6, 1),
                    rd.uniform(0.1, 0.4),
                    seed=None,
                    directed=False,
                )

    graph_class = [1 for i in range(50)] + [2 for i in range(50)]

    return graphs, graph_class


def synthetic_data_watts_strogatz(N=1000):

    graphs = []
    graph_labels = []

    p = np.linspace(0, 1, N)

    for i in range(N):
        G = nx.connected_watts_strogatz_graph(40, 5, p[i])
        graphs.append(G)
        graph_labels.append(p[i])

    return graphs, np.asarray(graph_labels)


def synthetic_data_powerlaw_cluster(N=1000):
    graphs = []
    graph_labels = []

    p = np.linspace(0, 1, N)

    for i in range(N):
        G = nx.powerlaw_cluster_graph(40, 5, p[i])
        graphs.append(G)
        graph_labels.append(p[i])

    return graphs, np.asarray(graph_labels)


def synthetic_data_sbm(N=1000):
    graphs = []
    graph_labels = []

    import random

    for i in range(int(N / 2)):
        G = nx.stochastic_block_model(
            [random.randint(10, 30), random.randint(10, 30), random.randint(10, 30)],
            [[0.6, 0.1, 0.1], [0.1, 0.6, 0.1], [0.1, 0.1, 0.6]],
        )
        graphs.append(G)
        graph_labels.append(1)

    for i in range(int(N / 2)):
        G = nx.stochastic_block_model(
            [random.randint(20, 40), random.randint(20, 40)], [[0.6, 0.1], [0.1, 0.6]]
        )
        graphs.append(G)
        graph_labels.append(2)

    return graphs, np.asarray(graph_labels)
