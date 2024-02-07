"""K Components class."""

from functools import lru_cache

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "KComponents"


@lru_cache(maxsize=None)
def eval_kcomponents(graph):
    """this evaluates the main function and cach it for speed up."""
    return nx.k_components(graph)


def max_num_components(graph):
    """max_num_components"""
    return max(len(i) for i in eval_kcomponents(graph).values())


def num_connectivity_levels_k(graph):
    """num_connectivity_levels_k"""
    return len(eval_kcomponents(graph).keys())


def size_max_k_component(graph):
    """size_max_k_component"""
    return len(eval_kcomponents(graph)[len(eval_kcomponents(graph).keys())][0])


def size_2_component(graph):
    """size_2_component"""
    return len(eval_kcomponents(graph)[2][0])


class KComponents(FeatureClass):
    """K Components class.

    Returns features based on the k component structure of a graph.

    A `k`-component is a maximal subgraph of a graph G that has, at least,
    node connectivity `k`: we need to remove at least `k` nodes to break it
    into more components. `k`-components have an inherent hierarchical
    structure because they are nested in terms of connectivity: a connected
    graph can contain several 2-components, each of which can contain
    one or more 3-components, and so forth.

    Uses networkx, see 'https://networkx.org/documentation/stable/reference/algorithms/\
        approximation.html`

    References
    ----------
    .. [1]  Torrents, J. and F. Ferraro (2015) Structural Cohesion:
            Visualization and Heuristics for Fast Computation.
            https://arxiv.org/pdf/1503.04476v1

    .. [2]  White, Douglas R., and Mark Newman (2001) A Fast Algorithm for
            Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
            http://eclectic.ss.uci.edu/~drwhite/working.pdf

    .. [3]  Moody, J. and D. White (2003). Social cohesion and embeddedness:
            A hierarchical conception of social groups.
            American Sociological Review 68(1), 103--28.
            http://www2.asanet.org/journals/ASRFeb03MoodyWhite.pdf

    """

    modes = ["slow"]
    shortname = "KC"
    name = "k_components"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "num_connectivity_levels_k",
            num_connectivity_levels_k,
            "The number of connectivity levels k in the input graphs",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "max_num_components",
            max_num_components,
            "The maximum number of componenets at any value of k",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "size_max_k_component",
            size_max_k_component,
            "The number of nodes of the component corresponding to the largest k",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "size_2_component",
            size_2_component,
            "The number of nodes in k=2 component",
            InterpretabilityScore(3),
        )
