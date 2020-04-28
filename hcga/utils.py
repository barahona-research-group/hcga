"""utils functions"""
import numpy as np

from hcga.graph import Graph


def get_trivial_graph(n_node_features=0):
    """Generate a trivial graph for internal purposes."""
    if n_node_features > 0:
        features = n_node_features * [0.0]
        nodes = [(0, features), (1, features), (2, features)]
    else:
        nodes = [0, 1, 2]
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 1, 1.0)]
    return Graph(nodes, edges, 0)


def filter_samples(features, sample_removal=0.05):
    """Remove samples with more than X% bad values."""
    samples_to_filter = features.index[
        features.isnull().sum(axis=1) / len(features.columns) > sample_removal
    ].tolist()
    features = features.drop(labels=samples_to_filter)
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
