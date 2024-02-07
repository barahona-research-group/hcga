"""Looplessness class."""

from functools import lru_cache, partial

import networkx as nx
import numpy as np
import sympy as sp

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Looplessness"


@lru_cache(maxsize=None)
def looplessness(graph):  # pylint: disable=too-many-locals
    """Looplessness measure class

    Implements features based on graph looplessness [1]_.

    If a graph is bipartite, the trophic levels of all nodes is 1:
        Proof:
            Let A be the adjacency matrix of g, a bipartite graph
            Basal nodes have trophic level = 1 by definition
            If node i in not a basal node, it will have out degree = 0
            Therefore row i of A will consist of zeros
            Trophic level formula:
                s_i = 1 + (1/k_i) * sum_j(A_ij*s_j)
                where:
                    s_i = trophic level of node i
                    k_i = in degree of node i
            For non basal node i, sum_j(A_ij*s_j) = 0, so s_i = 1
            Therefore the trophic level of all nodes is 1

    For a bipartite graph the branching factor, incoherence parameter, expected
    trophic coherence, expected branching factor and loop exponent are all 0:
        Proof:
            Nodes will either have 0 in degree or out degree. Results follows from this

    References
    ----------
    .. [1]Johnson, Samuel, and Nick S. Jones. "Looplessness in networks is linked to trophic\
        coherence.",
     Proceedings of the National Academy of Sciences 114.22 (2017): 5618-5623.

    """

    n = graph.number_of_nodes()

    # Bipartite graphs
    if nx.is_bipartite(graph) or not nx.is_directed(graph):
        trophic = [1.0] * n
        return 0, trophic, 0, 0, 0, 0

    # Non-bipartite graphs

    # Branching Factor
    in_degrees_dict = dict(graph.in_degree)  # Needed for trophic
    in_degrees = list(in_degrees_dict.values())
    out_degrees = list(dict(graph.out_degree).values())
    i_o = [j * k for j, k in zip(in_degrees, out_degrees)]
    mean_degree = graph.number_of_edges() / graph.number_of_nodes()
    branching_factor = np.mean(i_o) / mean_degree

    e = graph.number_of_edges()

    # Find basal nodes
    basal = [i for i in in_degrees_dict if in_degrees_dict[i] == 0]
    b_e = sum(out_degrees[i] for i in basal)

    # Compute expected trophic coherence and expected branching factor
    exp_trophic_coher = np.sqrt(e / b_e - 1)
    exp_branching_factor = (e - b_e) / (n - len(basal))

    # Compute trophic levels by solving system of equations
    # Basal nodes have trophic level 1 by definition
    t = list(range(n))
    non_basal = list(set(t).difference(set(basal)))

    trophic = [0] * n
    for i in basal:
        trophic[i] = 1

    s = sp.symbols(f"s0:{len(non_basal)}")

    for i, j in enumerate(non_basal):
        trophic[j] = s[i]

    # Convert all weights to 1 in order to compute trophic levels
    a = np.where(nx.adjacency_matrix(graph).toarray() > 0, 1, 0)

    LHS = [(tr - 1) * k for tr, k in zip(trophic, in_degrees)]
    RHS = list(np.dot(a, np.array(trophic)))

    # System of equations for non basal nodes
    equations = []
    for i in non_basal:
        equations.append(sp.Eq(LHS[i], RHS[i]))

    linear_solution = list(sp.linsolve(equations, s))

    sols = linear_solution[0]

    # Replace symbols with their trophic level value
    for i, j in enumerate(non_basal):
        trophic[j] = sols[i]

    trophic = [float(t) for t in trophic]

    # Compute trophic difference matrix
    trophic_diff = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            trophic_diff[i, j] = trophic[i] - trophic[j]
    trophic_diff_sq = np.square(trophic_diff)

    # Compute incoherence parameter
    incoherence_parameter = np.sqrt((np.sum(np.multiply(a, trophic_diff_sq)) - 1) / e)

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
    return looplessness(graph)[i]


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
