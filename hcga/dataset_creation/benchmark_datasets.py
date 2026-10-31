"""make benchmark datasets"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from hcga.graph import Graph, GraphCollection
from hcga.io import save_dataset


TU_URL = "https://www.chrsmrrs.com/graphkerneldatasets/"


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

    urllib.request.urlretrieve(f"{TU_URL}{dataset_name}.zip", f"{dataset_name}.zip")
    unzip(f"{dataset_name}.zip")

    graphs = extract_benchmark_graphs(dataset_name, dataset_name)
    save_dataset(graphs, dataset_name, folder=folder)

    shutil.rmtree(dataset_name)
    os.remove(f"{dataset_name}.zip")


def extract_benchmark_graphs(datadir, dataname):  # pylint: disable=too-many-locals
    """Read data from https://www.chrsmrrs.com/graphkerneldatasets/ (TU Dortmund format)
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    """
    prefix = str(Path(datadir) / dataname)

    with open(prefix + "_graph_indicator.txt") as f:
        nodes_df = pd.read_csv(f, dtype=int, header=None) - 1
    nodes_df.columns = ["graph_id"]

    with open(prefix + "_graph_labels.txt") as f:
        graph_labels = pd.read_csv(f, header=None)

    edges_df = pd.DataFrame()
    with open(prefix + "_A.txt") as f:
        for edges_df_next in pd.read_csv(f, sep=",", dtype=int, header=None, chunksize=10**6):
            edges_df = pd.concat([edges_df, edges_df_next - 1])
    edges_df.columns = ["start_node", "end_node"]
    edges_df["graph_id"] = nodes_df["graph_id"][edges_df["start_node"].to_list()].to_list()

    columns = []
    if Path(prefix + "_node_labels.txt").exists():
        with open(prefix + "_node_labels.txt") as f:
            nodes_df["labels_value"] = pd.read_csv(f, header=None)
        nodes_df["labels"] = list(pd.get_dummies(nodes_df["labels_value"]).to_numpy(dtype=float))
        columns.append("labels")

    if Path(prefix + "_node_attributes.txt").exists():
        with open(prefix + "_node_attributes.txt") as f:
            nodes_df["attributes"] = list(pd.read_csv(f, header=None).to_numpy())
        columns.append("attributes")

    graph_ids = list(set(nodes_df["graph_id"]))
    graphs = GraphCollection()
    for graph_id in graph_ids:
        nodes = nodes_df.loc[nodes_df["graph_id"] == graph_id][columns]
        edges = edges_df.loc[edges_df["graph_id"] == graph_id][["start_node", "end_node"]]
        graphs.add_graph(Graph(nodes, edges, int(graph_labels.loc[graph_id][0])))

    return graphs
