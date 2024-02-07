"""Vitality class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Vitality"


def vitality(graph):
    """"""
    return list(nx.closeness_vitality(graph).values())


class Vitality(FeatureClass):
    """Vitality measures class.

    Features based on the closeness vitality.
    The *closeness vitality* of a node, defined in Section 3.6.2 of [1],
    is the change in the sum of distances between all node pairs when
    excluding that node.

    References
    ----------
    .. [1] Ulrik Brandes, Thomas Erlebach (eds.).
           *Network Analysis: Methodological Foundations*.
           Springer, 2005.
           <http://books.google.com/books?id=TTNhSm7HYrIC>
    """

    modes = ["slow"]
    shortname = "V"
    name = "vitality"
    encoding = "networkx"

    def compute_features(self):
        # distribution of vitality
        self.add_feature(
            "vitality",
            vitality,
            "The closeness vitality of a node is the change in the sum of distances between \
            all node pairs when excluding that node",
            InterpretabilityScore(3),
            statistics="centrality",
        )
