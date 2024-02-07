"""Node clique number class."""

from networkx.algorithms import clique

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "NodeCliqueNumber"


def clique_sizes(graph):
    """clique_sizes"""
    return list(clique.node_clique_number(ensure_connected(graph)).values())


class NodeCliqueNumber(FeatureClass):
    """Node clique number class.

    Features based on the size of the largest maximal clique containing each node.

    Clique number calculations using networkx:
        `Networkx_clique <https://networkx.github.io/documentation/stable/reference/algorithms/\
            clique.html>`_
    """

    modes = ["fast", "medium", "slow"]
    shortname = "CN"
    name = "node_clique_number"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "clique_sizes",
            clique_sizes,
            "the distribution of clique sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )
