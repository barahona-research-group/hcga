"""K Components class."""
from functools import lru_cache

import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "KComponents"


class KComponents(FeatureClass):
    """K Components class."""

    modes = ["slow"]
    shortname = "KC"
    name = "k_components"
    encoding = "networkx"

    def compute_features(self):
        @lru_cache(maxsize=None)
        def eval_kcomponents(graph):
            """this evaluates the main function and cach it for speed up."""
            return nx.k_components(graph)

        self.add_feature(
            "num_connectivity_levels_k",
            lambda graph: len(eval_kcomponents(graph).keys()),
            "The number of connectivity levels k in the input graphs",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "max_num_components",
            lambda graph: max([len(i) for i in eval_kcomponents(graph).values()]),
            "The maximum number of componenets at any value of k",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "size_max_k_component",
            lambda graph: len(eval_kcomponents(graph)[len(eval_kcomponents(graph).keys())][0]),
            "The number of nodes of the component corresponding to the largest k",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "size_2_component",
            lambda graph: len(eval_kcomponents(graph)[2][0]),
            "The number of nodes in k=2 component",
            InterpretabilityScore(3),
        )
