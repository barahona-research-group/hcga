"""make synthetic datasets: WIP!!"""
import os
import re
import shutil
import zipfile

import networkx as nx
import numpy as np
import wget

from hcga.io import save_dataset
from hcga.graph import Graph, GraphCollection


def make(
    folder="./datasets", write_to_file=True, graph_type="SBM",
):
    if graph_type == "SBM":
        graphs = make_SBM()

    if write_to_file:
        save_dataset(graphs, "SYNTH_" + graph_type, folder=folder)


def make_SBM():
    """Make SBM with 1, 2, 3 and 4 clusters."""

    def _add_graph(label):
        for _ in range(50):
            graph = nx.stochastic_block_model(sizes, probs)
            graphs.add_graph(Graph(list(graph.nodes), list(graph.edges), label))

    graphs = GraphCollection()

    sizes = [10, 10, 10, 10]
    probs = [
        [0.2, 0.05, 0.05, 0.05],
        [0.05, 0.2, 0.05, 0.05],
        [0.05, 0.05, 0.2, 0.05],
        [0.05, 0.05, 0.05, 0.2],
    ]
    _add_graph(4)

    sizes = [13, 13, 14]
    probs = [[0.2, 0.05, 0.05], [0.05, 0.2, 0.05], [0.05, 0.05, 0.2]]
    _add_graph(3)

    sizes = [20, 20]
    probs = [[0.2, 0.05], [0.05, 0.2]]
    _add_graph(2)

    sizes = [40]
    probs = [[0.2]]
    _add_graph(1)

    return graphs


### below are deprecated functions ###
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
