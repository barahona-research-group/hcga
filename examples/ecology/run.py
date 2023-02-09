from git import Repo
from tqdm import tqdm
import networkx as nx
import numpy as np
from pathlib import Path
import pandas as pd

from hcga.graph import Graph, GraphCollection
from hcga.io import save_dataset
from hcga.extraction import extract
from hcga.io import load_dataset
from hcga.io import save_features
from hcga.io import load_features


def make_graph(data, label):
    edgelist = list(zip(data[0], data[1]))
    g = nx.Graph()
    g.add_edges_from(edgelist)
    sA = nx.to_scipy_sparse_array(g)

    r, c = sA.nonzero()
    edges = np.array([r, c, sA.data]).T

    # passing edge list to pandas dataframe
    edges_df = pd.DataFrame(edges, columns=["start_node", "end_node", "weight"])

    # creating node ids based on size of adjancency matrix
    nodes = np.arange(0, g.number_of_nodes())

    # loading node ids into dataframe
    nodes_df = pd.DataFrame(index=nodes)

    # create a single graph object
    return Graph(nodes_df, edges_df, label)


if __name__ == "__main__":
    if not Path("dataset").exists():
        Repo.clone_from("git@github.com:mjsmith037/Classifying_Bipartite_Networks.git", "dataset")

    if not Path("graphs.pkl").exists():
        graphs = GraphCollection()
        df = pd.read_csv("dataset/Data/Metadata.csv")

        for i, tpe in enumerate(df["type"].unique()):
            print(f"importing {tpe}")
            for name in tqdm(df[df["type"] == tpe].name):
                try:
                    data = pd.read_csv(f"dataset/Data/edgelists/{tpe}/{name}.csv", header=None)
                    graphs.add_graph(make_graph(data, i))
                except FileNotFoundError:
                    pass
        print(graphs)
        save_dataset(graphs, "graphs", folder=".")
    else:
        graphs = load_dataset("graphs.pkl")

    if not Path("features.pkl").exists():
        features, features_info = extract(
            graphs,
            n_workers=8,
            mode="fast",
            statistics_level="basic",
            with_runtimes=False,
            with_node_features=False,
            timeout=5,
        )

        save_features(features, features_info, graphs, "features.pkl")
    else:
        features, features_info, graphs = load_features(filename='features.pkl')
