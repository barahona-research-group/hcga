"""make benchmark datasets"""
from pathlib import Path
import os
import re
import shutil
import zipfile
from collections import defaultdict

import networkx as nx
import numpy as np
import wget

from ..io import save_dataset
from hcga.graph import Graph, GraphCollection


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
    unzip("{}.zip".format(dataset_name))

    graphs = extract_benchmark_graphs(dataset_name, dataset_name)

    save_dataset(graphs, dataset_name, folder=folder)

    #shutil.rmtree(dataset_name)
    #os.remove("{}.zip".format(dataset_name))
    print('\n')


def extract_benchmark_graphs(datadir, dataname, max_nodes=None):
    """ Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    """
    prefix = str(Path(datadir) / dataname)
    
    with open(prefix + "_graph_indicator.txt") as f:
        graph_indic = np.loadtxt(f, dtype=np.int)
    node_graph_ids = {i: graph_id - 1 for i, graph_id in enumerate(graph_indic)}

    with open( prefix + "_node_labels.txt") as f:
        node_labels = np.loadtxt(f, dtype=np.int)

    with open(prefix + "_node_attributes.txt") as f:
        node_attributes = np.loadtxt(f, delimiter=', ', dtype=np.float)

    with open(prefix + "_graph_labels.txt") as f:
        graph_labels = np.loadtxt(f)

    with open(prefix + "_A.txt") as f:
        adj_list = np.loadtxt(f, delimiter=', ', dtype=np.int)

    edge_list = defaultdict(list)
    for edge in adj_list:
        edge_list[node_graph_ids[edge[0] - 1]].append(tuple(edge))

    node_list = defaultdict(list)
    for (node, graph_id), node_attribute in zip(node_graph_ids.items(), node_attributes):
        if len(np.shape(node_attribute)) == 0:
            node_attribute = [node_attribute]
        else:
            node_attribute = list(node_attribute)
        node_list[graph_id].append(tuple([node, node_attribute]))

    graphs = GraphCollection()
    for graph_id in edge_list:
       graphs.add_graph(Graph(node_list[graph_id], edge_list[graph_id], graph_labels[graph_id]))

    return graphs
