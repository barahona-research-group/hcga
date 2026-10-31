"""Looplessness class."""

from functools import lru_cache, partial

import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Looplessness"


@lru_cache(maxsize=None)
def looplessness(graph):  # pylint: disable=too-many-locals
    """Looplessness measures of a directed graph [1]_.

    Trophic levels s solve s_i = 1 + (1/k_i^in) sum_j a_ji s_j, with s = 1 for basal nodes
    (in-degree 0), where a_ji = 1 for an edge j -> i (edge weights are ignored). The trophic
    incoherence parameter q is the standard deviation of the trophic differences s_i - s_j
    over the edges j -> i. Both come from networkx (``trophic_levels`` and
    ``trophic_incoherence_parameter``).

    Returns None for undirected graphs (trophic levels need edge directions), and for directed
    graphs without basal nodes or with nodes not reachable from a basal node (the trophic levels
    are then undefined).

    References
    ----------
    .. [1]Johnson, Samuel, and Nick S. Jones. "Looplessness in networks is linked to trophic\
        coherence.",
     Proceedings of the National Academy of Sciences 114.22 (2017): 5618-5623.

    """
    if not graph.is_directed():
        return None

    n = graph.number_of_nodes()
    e = graph.number_of_edges()
    in_degrees = np.array([d for _, d in graph.in_degree])
    out_degrees = np.array([d for _, d in graph.out_degree])
    basal = in_degrees == 0
    b_e = out_degrees[basal].sum()
    if e == 0 or b_e == 0:
        return None

    try:
        trophic_levels = nx.trophic_levels(graph, weight=None)
    except (nx.NetworkXError, np.linalg.LinAlgError):  # a node unreachable from basal nodes
        return None
    trophic = [float(trophic_levels[node]) for node in graph]

    # Branching Factor
    branching_factor = np.mean(in_degrees * out_degrees) / (e / n)

    # Compute expected trophic coherence and expected branching factor
    exp_trophic_coher = np.sqrt(e / b_e - 1)
    exp_branching_factor = (e - b_e) / (n - basal.sum())

    # Incoherence parameter
    incoherence_parameter = np.float64(nx.trophic_incoherence_parameter(graph, weight=None))

    # Compute loop exponent
    loop_exponent = (
        np.log(branching_factor)
        + 1 / (2 * (exp_trophic_coher**2))
        - 1 / (2 * (incoherence_parameter**2))
    )

    return (
        branching_factor,
        trophic,
        incoherence_parameter,
        exp_trophic_coher,
        exp_branching_factor,
        loop_exponent,
    )


def _get(graph, i):
    """get output"""
    result = looplessness(graph)
    if result is None:
        return None
    return result[i]


class Looplessness(FeatureClass):
    """Looplessness class."""

    modes = ["fast", "medium", "slow"]
    shortname = "Lln"
    name = "looplessness"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "branching_factor",
            partial(_get, i=0),
            "The branching factor of the graph",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "trophic_levels",
            partial(_get, i=1),
            "The distribution of trophic levels",
            InterpretabilityScore(5),
            statistics="centrality",
        )

        self.add_feature(
            "incoherence_parameter",
            partial(_get, i=2),
            "The trophic incoherence parameter of the graph",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "exp_trophic_coherence",
            partial(_get, i=3),
            "The expected trophic incoherence parameter of the graph",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "exp_branching_factor",
            partial(_get, i=4),
            "The expected branching factor of the graph",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "loop_exponent",
            partial(_get, i=5),
            "The loop exponent of the graph",
            InterpretabilityScore(5),
        )
