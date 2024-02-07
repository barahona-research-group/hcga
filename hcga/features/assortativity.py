"""Assortativity class."""

from functools import partial

import networkx as nx
from networkx.algorithms import assortativity

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Assortativity"


def average_neighbor_degree(graph):
    """Average neighbor degree."""
    return list(assortativity.average_neighbor_degree(graph).values())


def weigted_average_neighbor_degree(graph):
    """Average neighbor degree."""
    return list(assortativity.average_neighbor_degree(graph, weight="weight").values())


class Assortativity(FeatureClass):
    """Assortativity class.

    Features derived from different measures of assortativity of the graph.

    Uses networkx, see `https://networkx.github.io/documentation/networkx-2.4/reference/\
        algorithms/assortativity.html`

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    .. [2] Foster, J.G., Foster, D.V., Grassberger, P. & Paczuski, M.
       Edge direction and the structure of networks, PNAS 107, 10815-20 (2010).


    """

    modes = ["fast", "medium", "slow"]
    shortname = "AS"
    name = "assortativity"
    encoding = "networkx"

    def compute_features(self):
        # Adding basic node and edge numbers
        self.add_feature(
            "degree_assortativity_coeff",
            nx.degree_assortativity_coefficient,
            "Similarity of connections in the graph with respect to the node degree",
            InterpretabilityScore(4),
        )
        self.add_feature(
            "degree_assortativity_coeff_weighted",
            partial(nx.degree_assortativity_coefficient, weight="weight"),
            "Similarity of connections in the graph with respect to the node degree",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "degree_assortativity_coeff_pearson",
            nx.degree_pearson_correlation_coefficient,
            "Similarity of connections in the graph with respect to the node degree",
            InterpretabilityScore(4),
        )
        self.add_feature(
            "degree_assortativity_coeff_pearson_weighted",
            partial(nx.degree_pearson_correlation_coefficient, weight="weight"),
            "Similarity of connections in the graph with respect to the node degree",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "degree assortativity",
            average_neighbor_degree,
            "average neighbor degree",
            InterpretabilityScore(4),
            statistics="centrality",
        )
        self.add_feature(
            "degree assortativity_weighted",
            weigted_average_neighbor_degree,
            "average neighbor degree",
            InterpretabilityScore(4),
            statistics="centrality",
        )
