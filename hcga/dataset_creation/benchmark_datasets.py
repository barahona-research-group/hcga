"""make benchmark datasets"""
import os
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import wget

from hcga.graph import Graph, GraphCollection

from ..io import save_dataset


def unzip(zip_filename):
    """unzip a file"""
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall()


def make(dataset_name="ENZYMES", folder="./datasets"):
    """
    Standard datasets include:
        DD
        ENZYMES
        REDDIT-MULTI-12K
        PROTEINS
        MUTAG
    """

    wget.download(
        "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{}.zip".format(
            dataset_name
        )
    )
    print("\n")
    unzip("{}.zip".format(dataset_name))

    graphs = extract_benchmark_graphs(dataset_name, dataset_name)
    save_dataset(graphs, dataset_name, folder=folder)

    shutil.rmtree(dataset_name)
    os.remove("{}.zip".format(dataset_name))


def extract_benchmark_graphs(datadir, dataname, max_nodes=None):
    """ Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    """
    prefix = str(Path(datadir) / dataname)

    with open(prefix + "_graph_indicator.txt") as f:
        graph_indic = pd.read_csv(f, dtype=np.int, header=None).to_numpy().flatten()
    node_graph_ids = {i: graph_id - 1 for i, graph_id in enumerate(graph_indic)}

    with open(prefix + "_graph_labels.txt") as f:
        graph_labels = pd.read_csv(f, header=None).to_numpy().flatten()

    with open(prefix + "_A.txt") as f:
        adj_list = pd.read_csv(
            f, sep=", ", delimiter=None, dtype=np.int, header=None
        ).to_numpy()  # .flatten()

    edge_list = defaultdict(list)
    for edge in adj_list:
        edge_list[node_graph_ids[edge[0] - 1]].append(tuple(edge - 1))

    with open(prefix + "_node_labels.txt") as f:
        node_labels = pd.read_csv(f, header=None).to_numpy().flatten()

    if Path(prefix + "_node_attributes.txt").exists():
        with open(prefix + "_node_attributes.txt") as f:
            node_attributes = pd.read_csv(f, header=None).to_numpy()
        node_features = [
            [node_label] + list(node_attribute)
            for node_label, node_attribute in zip(node_labels, node_attributes)
        ]
    else:
        node_features = [[node_label] for node_label in node_labels]

    node_list = defaultdict(list)
    for (node, graph_id), node_feature in zip(node_graph_ids.items(), node_features):
        node_list[graph_id].append(tuple([node, node_feature]))

    graphs = GraphCollection()
    for graph_id in edge_list:
        graphs.add_graph(
            Graph(node_list[graph_id], edge_list[graph_id], graph_labels[graph_id])
        )

    return graphs
