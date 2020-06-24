"""Utils functions."""
import logging
import numpy as np
import pandas as pd

from hcga.graph import Graph

L = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Timeout exception."""


def timeout_handler(signum, frame):
    """Function to raise timeout exception."""
    raise TimeoutError


def get_trivial_graph(n_node_features=0):
    """Generate a trivial graph for internal purposes."""
    nodes = pd.DataFrame([0, 1, 2])
    if n_node_features > 0:
        nodes["features"] = 3 * [n_node_features * [0.0]]
    edges = pd.DataFrame()
    edges["start_node"] = [0, 1, 2]
    edges["end_node"] = [1, 2, 0]
    return Graph(nodes, edges, 0)


def filter_graphs(features, graph_removal=0.05):
    """Remove samples with more than X% bad values."""
    samples_to_filter = features.index[
        features.isnull().sum(axis=1) / len(features.columns) > graph_removal
    ].tolist()
    features = features.drop(labels=samples_to_filter)
    L.info(
        "%s graphs were removed for more than %s fraction of bad features",
        str(len(samples_to_filter)),
        str(graph_removal),
    )
    return features


def filter_features(features):
    """Filter features and create feature matrix."""
    # remove inf and nan
    nan_features = features.replace([np.inf, -np.inf], np.nan)

    # remove features with nans
    valid_features = nan_features.dropna(axis=1)

    # remove features with equal values accros graphs
    return valid_features.drop(
        valid_features.std()[(valid_features.std() == 0)].index, axis=1
    )
