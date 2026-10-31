"""make synthetic datasets: WIP!!"""

import random

import networkx as nx
import numpy as np

from hcga.graph import GraphCollection
from hcga.io import save_dataset

SEED = 0


def _graph_seed(rng):
    """Draw an integer seed for one networkx generator call."""
    return int(rng.integers(2**31))


def make(folder="./datasets", write_to_file=True, graph_type="SBM", seed=SEED):
    """Make dataset (reproducible for a given seed), and return it."""
    if graph_type == "SBM":
        graphs = make_SBM(seed=seed)
    else:
        raise Exception("This type of synthetic dataset is not yet implemented")
    if write_to_file:
        save_dataset(graphs, "SYNTH_" + graph_type, folder=folder)
    return graphs


def make_SBM(seed=SEED):
    """Make SBM with 1, 2, 3 and 4 clusters."""
    n_graphs = 10
    rng = np.random.default_rng(seed)

    def _add_graph(label):
        for _ in range(n_graphs):
            graph = nx.stochastic_block_model(sizes, probs, seed=_graph_seed(rng))
            graphs.add_graph(graph, label=label)

    graphs = GraphCollection()

    sizes = [20, 20, 20, 20]
    probs = [
        [0.5, 0.02, 0.02, 0.02],
        [0.02, 0.5, 0.02, 0.02],
        [0.02, 0.02, 0.5, 0.02],
        [0.02, 0.02, 0.02, 0.5],
    ]
    _add_graph(4)

    sizes = [27, 27, 26]
    probs = [[0.5, 0.02, 0.02], [0.02, 0.5, 0.02], [0.02, 0.02, 0.5]]
    _add_graph(3)

    sizes = [40, 40]
    probs = [[0.5, 0.02], [0.02, 0.5]]
    _add_graph(2)

    sizes = [80]
    probs = [[0.5]]
    _add_graph(1)

    return graphs


# below are deprecated functions
def synthetic_data_watts_strogatz(N=1000, seed=SEED):
    """small world."""
    graphs = []
    graph_labels = []
    rng = np.random.default_rng(seed)

    p = np.linspace(0, 1, N)

    for i in range(N):
        G = nx.connected_watts_strogatz_graph(40, 5, p[i], seed=_graph_seed(rng))
        graphs.append(G)
        graph_labels.append(p[i])

    return graphs, np.asarray(graph_labels)


def synthetic_data_powerlaw_cluster(N=1000, seed=SEED):
    """powerlaw."""
    graphs = []
    graph_labels = []
    rng = np.random.default_rng(seed)

    p = np.linspace(0, 1, N)

    for i in range(N):
        G = nx.powerlaw_cluster_graph(40, 5, p[i], seed=_graph_seed(rng))
        graphs.append(G)
        graph_labels.append(p[i])

    return graphs, np.asarray(graph_labels)


def synthetic_data_sbm(N=1000, seed=SEED):
    """sbm"""
    graphs = []
    graph_labels = []
    rand = random.Random(seed)

    for _ in range(int(N / 2)):
        G = nx.stochastic_block_model(
            [rand.randint(10, 30), rand.randint(10, 30), rand.randint(10, 30)],
            [[0.6, 0.1, 0.1], [0.1, 0.6, 0.1], [0.1, 0.1, 0.6]],
            seed=rand.randrange(2**31),
        )
        graphs.append(G)
        graph_labels.append(1)

    for _ in range(int(N / 2)):
        G = nx.stochastic_block_model(
            [rand.randint(20, 40), rand.randint(20, 40)],
            [[0.6, 0.1], [0.1, 0.6]],
            seed=rand.randrange(2**31),
        )
        graphs.append(G)
        graph_labels.append(2)

    return graphs, np.asarray(graph_labels)
