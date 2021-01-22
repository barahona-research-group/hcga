"""Node clique number class."""
from networkx.algorithms import clique

from ..feature_class import FeatureClass, InterpretabilityScore
from .utils import ensure_connected

featureclass_name = "NodeCliqueNumber"


class NodeCliqueNumber(FeatureClass):
    """Node clique number class.

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
            "clique sizes",
            lambda graph: list(clique.node_clique_number(ensure_connected(graph)).values()),
            "the distribution of clique sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )
