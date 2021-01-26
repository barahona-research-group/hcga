"""Node Features class."""
from functools import lru_cache

import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "NodeFeatures"


class NodeFeatures(FeatureClass):
    """Node Features class.

    Features based on node features.

    Here, we use the node features in various ways to compute new features.

    1. Aggregate node features (ignores graph structure).
    2. Convolute node features using message-passing, then aggregate (incorporates graph structure).

    Graph convolutions follow a similar pattern to that of graph neural networks [1]_.

    References
    ----------
    .. [1] Robert L. Peach,Alexis Arnaudon,Mauricio Barahona.
           Semi-supervised classification on graphs using explicit diffusion dynamics,
           Foundations of Data Science,2,1,19,33,2020-2-11,

    """

    modes = ["fast", "medium", "slow"]
    shortname = "NF"
    name = "node_features"
    encoding = "networkx"

    def compute_features(self):
        @lru_cache(maxsize=None)
        def get_feature_matrix(graph):
            """Extracting feature matrix."""
            return np.vstack([graph.nodes[node]["feat"] for node in graph.nodes])

        @lru_cache(maxsize=None)
        def get_conv_matrix(graph):
            """Extracting feature matrix."""
            return nx.to_numpy_array(graph) + np.eye(len(graph))

        self.add_feature(
            "node_feature",
            lambda graph: get_feature_matrix(graph),
            "The summary statistics of node feature ",
            InterpretabilityScore(5),
            statistics="node_features",
        )

        self.add_feature(
            "conv_node_feature",
            lambda graph: get_conv_matrix(graph).dot(get_feature_matrix(graph)),
            "The summary statistics after a single message passing of features of node feature ",
            InterpretabilityScore(3),
            statistics="node_features",
        )

        self.add_feature(
            "conv2_node_feature",
            lambda graph: np.linalg.matrix_power(get_conv_matrix(graph), 2).dot(
                get_feature_matrix(graph)
            ),
            "The summary statistics after a two message passing of features of node feature ",
            InterpretabilityScore(3),
            statistics="node_features",
        )

        self.add_feature(
            "mean_node_features_featurewise",
            lambda graph: np.mean(get_feature_matrix(graph), axis=1).tolist(),
            "The summary statistics of the mean of node features ",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "mean_node_feature_nodewise",
            lambda graph: np.mean(get_feature_matrix(graph), axis=0).tolist(),
            "The summary statistics of the mean of each feature across nodes ",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "max_node_feature_featurewise",
            lambda graph: np.max(get_feature_matrix(graph), axis=1).tolist(),
            "The summary statistics of the max of node features ",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "max_node_feature_nodewise",
            lambda graph: np.max(get_feature_matrix(graph), axis=0).tolist(),
            "The summary statistics of the max of each feature across nodes ",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "min_node_feature_featurewise",
            lambda graph: np.min(get_feature_matrix(graph), axis=1).tolist(),
            "The summary statistics of the min of node features ",
            InterpretabilityScore(3),
            statistics="centrality",
        )
        self.add_feature(
            "min_node_feature_nodewise",
            lambda graph: np.min(get_feature_matrix(graph), axis=0).tolist(),
            "The summary statistics of the min of each feature across nodes ",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "sum_node_feature_featurewise",
            lambda graph: np.sum(get_feature_matrix(graph), axis=1).tolist(),
            "The summary statistics of the sum of node features ",
            InterpretabilityScore(3),
            statistics="centrality",
        )
        self.add_feature(
            "sum_node_feature_nodewise",
            lambda graph: np.sum(get_feature_matrix(graph), axis=0).tolist(),
            "The summary statistics of the sum of each feature across nodes ",
            InterpretabilityScore(3),
            statistics="centrality",
        )
