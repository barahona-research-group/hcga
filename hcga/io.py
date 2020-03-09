"""input/output functions"""
import sys
import os
import pickle
import csv
import numpy as np
from pathlib import Path


def _ensure_weights(graphs):
    """ensure that graphs edges have a weights value"""
    for i, graph in enumerate(graphs):
        for u, v in graph.edges:
            if "weight" not in graph[u][v]:
                graph[u][v]["weight"] = 1.0


def _remove_small_graphs(graphs, n_node_min=2):
    """remove too small graphs"""
    return [graph for graph in graphs if len(graph) > n_node_min]


def save_analysis(
    X, testing_accuracy, top_features, folder="/", filename="sklearn_analysis"
):
    """save results of analysis"""
    with open(os.path.join(folder, filename + ".pkl"), "wb") as f:
        pickle.dump([X, testing_accuracy, top_features], f)


def load_analysis(folder="/", filename="sklearn_analysis"):
    """save results of analysis"""
    with open(os.path.join(folder, filename + ".pkl"), "rb") as f:
        return pickle.load(f)


def save_dataset(graphs, labels, filename, folder="./datasets"):
    """Save a dataset in a pickle"""
    if not os.path.exists(folder):
        os.mkdir(folder)

    with open(os.path.join(folder, filename + ".pkl"), "wb") as f:
        pickle.dump([graphs, labels], f)


def load_dataset(filename):
    """load a dataset from a pickle"""
    with open(filename, "rb") as f:
        graphs_full, labels = pickle.load(f)

    graphs = []
    for i in range(len(graphs_full)):
        graph = graphs_full[i]
        graph.label = labels[i]
        graphs.append(graph)

    graphs = _remove_small_graphs(graphs)
    _ensure_weights(graphs)

    return graphs


def save_features(feature_matrix, feature_info, filename="./features.pkl"):
    """Save the features in a pickle"""
    if not Path(filename).parent.is_dir():
        Path(filename).parent.mkdir()

    pickle.dump(
        [feature_matrix, feature_info],
        open(filename, "wb"),
    )


def load_features(filename="features.pkl"):
    """Save the features in a pickle"""
    return pickle.load(open(filename, "rb"))
