"""input/output functions."""

import os
import pickle
from pathlib import Path

import numpy as np


def _ensure_weights(graph):
    """ensure that graphs edges have a weights value."""
    for u, v in graph.edges:
        if "weight" not in graph[u][v]:
            graph[u][v]["weight"] = 1.0


def _set_node_features(graph):
    """If no node features, set it to 0."""
    for node in graph.nodes:
        if "feat" not in graph.nodes[node]:
            graph.nodes[node]["feat"] = [0]

        feat_shape = np.shape(graph.nodes[node]["feat"])
        if not feat_shape:
            graph.nodes[node]["feat"] = [graph.nodes[node]["feat"]]
        elif len(feat_shape) > 1:
            raise Exception("Please provide flat node features vector.")


def save_analysis(X, explainer, shap_values, folder=".", filename="analysis_results"):
    """save results of analysis."""
    if not Path(folder).exists():
        os.mkdir(folder)

    with open(os.path.join(folder, filename + ".pkl"), "wb") as f:
        pickle.dump([X, explainer, shap_values], f)


def load_analysis(folder=".", filename="analysis_results"):
    """Load results of analysis."""
    with open(os.path.join(folder, filename + ".pkl"), "rb") as f:
        return pickle.load(f)


def save_dataset(graphs, filename, folder="./datasets"):
    """Save a dataset in a pickle."""
    if not os.path.exists(folder):
        os.mkdir(folder)

    with open(os.path.join(folder, filename + ".pkl"), "wb") as f:
        pickle.dump(graphs, f)


def load_dataset(filename):
    """Load a dataset from a pickle."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_features(features, feature_info, graphs, filename="./features.pkl"):
    """Save the features in a pickle."""
    if not Path(filename).parent.is_dir():
        Path(filename).parent.mkdir()
    with open(filename, "wb") as f:
        pickle.dump([features, feature_info, graphs], f)


def load_features(filename="features.pkl"):
    """Load the features from a pickle."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def _model_path(folder, filename):
    """Path of a model pickle, adding the .pkl extension unless filename already has it."""
    path = Path(folder) / filename
    if path.suffix != ".pkl":
        path = path.with_name(path.name + ".pkl")
    return path


def save_fitted_model(fitted_model, scaler, feature_info, folder=".", filename="./model"):
    """Save a fitted model in a pickle."""
    path = _model_path(folder, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump([fitted_model, scaler, feature_info], f)


def load_fitted_model(folder=".", filename="model"):
    """Load a fitted model from a pickle."""
    with open(_model_path(folder, filename), "rb") as f:
        return pickle.load(f)
