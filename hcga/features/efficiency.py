"""Efficiency class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Efficiency"


def _undirected(graph):
    """Efficiencies are defined for undirected graphs, so digraphs are made undirected."""
    if graph.is_directed():
        return graph.to_undirected()
    return graph


def local_efficiency(graph):
    """local_efficiency"""
    return nx.local_efficiency(_undirected(graph))


def global_efficiency(graph):
    """global_efficiency"""
    return nx.global_efficiency(_undirected(graph))


class Efficiency(FeatureClass):
    """EFficiency Measures class.

    Features based on the efficiency of a gaph.
    The *efficiency* of a pair of nodes is the multiplicative inverse of the
    shortest path distance between the nodes [1]_. Returns 0 if no path
    between nodes.

    Uses networkx: `Efficiency Measures <https://networkx.org/documentation/stable/reference/\
        algorithms/efficiency_measures.html>`_

    References
    ----------
    .. [1] Latora, Vito, and Massimo Marchiori.
           "Efficient behavior of small-world networks."
           *Physical Review Letters* 87.19 (2001): 198701.
           <https://doi.org/10.1103/PhysRevLett.87.198701>

    """

    modes = ["fast", "medium", "slow"]
    shortname = "EF"
    name = "efficiency"
    encoding = "networkx"

    def compute_features(self):
        # local effiency
        self.add_feature(
            "local_efficiency",
            local_efficiency,
            "The local efficiency",
            InterpretabilityScore(4),
        )

        # global effiency
        self.add_feature(
            "global_efficiency",
            global_efficiency,
            "The global efficiency",
            InterpretabilityScore(4),
        )
