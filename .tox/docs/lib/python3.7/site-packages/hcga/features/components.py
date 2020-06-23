"""Components class."""
from functools import lru_cache
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Components"


class Components(FeatureClass):
    """Components class.

    Components calculations using network: `Networkx_components <https://networkx.github.io/\
        documentation/stable/reference/algorithms/component.html>`_
    """

    modes = ["fast", "medium", "slow"]
    shortname = "C"
    name = "components"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "is_connected",
            lambda graph: nx.is_connected(graph) * 1,
            "Whether the graph is connected or not",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "num_connected_components",
            lambda graph: len(list(nx.connected_components(graph))),
            "The number of connected components",
            InterpretabilityScore(5),
        )

        @lru_cache(maxsize=None)
        def eval_connectedcomponents(graph):
            """this evaluates the main function and cach it for speed up."""
            return list(nx.connected_components(graph))

        self.add_feature(
            "largest_connected_component",
            lambda graph: len(eval_connectedcomponents(graph)[0]),
            "The size of the largest connected component",
            InterpretabilityScore(4),
        )

        def ratio_largest(graph):
            if len(eval_connectedcomponents(graph)) == 1:
                return 0
            return len(eval_connectedcomponents(graph)[0]) / len(
                eval_connectedcomponents(graph)[1]
            )

        self.add_feature(
            "ratio_largest_connected_components",
            ratio_largest,
            "The size ratio of the two largest connected components",
            InterpretabilityScore(4),
        )

        def ratio_min_max(graph):
            if len(eval_connectedcomponents(graph)) == 1:
                return 0
            return len(eval_connectedcomponents(graph)[0]) / len(
                eval_connectedcomponents(graph)[-1]
            )

        self.add_feature(
            "ratio_maxmin_connected_components",
            ratio_min_max,
            "The size ratio of the max and min largest connected components",
            InterpretabilityScore(4),
        )
