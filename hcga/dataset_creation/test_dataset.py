"""make test datasets"""

import networkx as nx
import numpy as np
import pandas as pd

from hcga.graph import Graph, GraphCollection
from hcga.io import save_dataset

SEED = 0


def _add_graph_desc(g, desc):
    """Add descrition desc to graph g as a graph attribute"""
    g.graph["description"] = desc
    return g


def add_dummy_node_features(graph, rng=None):
    """add random node features"""
    if rng is None:
        rng = np.random.default_rng(SEED)
    for u in graph.nodes():
        graph.nodes[u]["feat"] = rng.random(10)
    return graph


def make(folder="./datasets", add_features=False, write_to_file=True, n_graphs=5, seed=SEED):
    """Makes pickle with graphs that test robustness of hcga"""
    rng = np.random.default_rng(seed)

    graphs = []
    # one, two and three node graphs
    for _ in range(n_graphs):
        graphs.append(_add_graph_desc(nx.grid_graph([1]).copy(), "one-node graph"))
        graphs.append(_add_graph_desc(nx.grid_graph([2]).copy(), "two-node graph"))
        graphs.append(_add_graph_desc(nx.grid_graph([3]).copy(), "three-node graph"))

    # no edges
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1, weight=2)
    G.add_node(2, weight=3)
    for _ in range(n_graphs):
        graphs.append(_add_graph_desc(G.copy(), "graph without edges"))

    # directed graph no weights
    G = nx.DiGraph()
    G.add_nodes_from(range(100, 110))
    for _ in range(n_graphs):
        graphs.append(_add_graph_desc(G.copy(), "directed graph with no weights"))

    # directed graph weighted
    G = nx.DiGraph()
    H = nx.path_graph(10)
    G.add_nodes_from(H)
    G.add_edges_from(H.edges)
    for _ in range(n_graphs):
        graphs.append(_add_graph_desc(G.copy(), "directed graph weighted"))

    # adding features to all
    if add_features:
        graphs = [add_dummy_node_features(graph, rng) for graph in graphs]

    graphs_coll = GraphCollection()
    for graph in graphs:
        nodes = pd.DataFrame(list(graph.nodes))
        edges = pd.DataFrame(columns=["start_node", "end_node"])
        for i, edge in enumerate(graph.edges):
            edges.loc[i, "start_node"] = edge[0]
            edges.loc[i, "end_node"] = edge[1]
        graph_type = "directed" if graph.is_directed() else None
        graphs_coll.add_graph(Graph(nodes, edges, int(rng.integers(0, 2)), graph_type=graph_type))

    if write_to_file:
        save_dataset(graphs_coll, "TESTDATA", folder=folder)

    return graphs_coll
