"""make test datasets"""

import networkx as nx
import numpy as np
import pandas as pd

from hcga.graph import Graph, GraphCollection
from hcga.io import save_dataset


def _add_graph_desc(g, desc):
    """Add descrition desc to graph g as a graph attribute"""
    g.graph["description"] = desc
    return g


def add_dummy_node_features(graph):
    """add random node features"""
    for u in graph.nodes():
        graph.nodes[u]["feat"] = np.random.rand(10)
    return graph


def make(folder="./datasets", add_features=False, write_to_file=True, n_graphs=5):
    """Makes pickle with graphs that test robustness of hcga"""

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
        graphs = [add_dummy_node_features(graph) for graph in graphs]

    graphs_coll = GraphCollection()
    for graph in graphs:
        nodes = pd.DataFrame(list(graph.nodes))
        edges = pd.DataFrame(columns=["start_node", "end_node"])
        for i, edge in enumerate(graph.edges):
            edges.loc[i, "start_node"] = edge[0]
            edges.loc[i, "end_node"] = edge[1]
        graphs_coll.add_graph(Graph(nodes, edges, np.random.randint(0, 2)))

    if write_to_file:
        save_dataset(graphs_coll, "TESTDATA", folder=folder)

    return graphs_coll
