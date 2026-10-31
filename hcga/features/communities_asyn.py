"""Communities Asyn class."""

from functools import lru_cache, partial

import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "CommunitiesAsyn"

SEED = 0


@lru_cache(maxsize=None)
def eval_asyn(graph, num_comms):
    """Fluid communities of the largest connected component, sorted by size (largest first).

    Directed graphs are treated as undirected.
    """
    connected_graph = ensure_connected(graph)
    if connected_graph.is_directed():
        connected_graph = nx.Graph(connected_graph)
    communities = [
        set(c) for c in nx.community.asyn_fluidc(connected_graph, int(num_comms), seed=SEED)
    ]
    return sorted(communities, key=len, reverse=True)


def _densities(graph, num_comms):
    """Fluid density 1/|C| of each fluid community."""
    return [1.0 / len(c) for c in eval_asyn(graph, num_comms)]


def sum_density(graph, num_comms):
    """sum_density"""
    return sum(_densities(graph, num_comms))


def ratio_density(graph, num_comms):
    """ratio_density"""
    return np.min(_densities(graph, num_comms)) / np.max(_densities(graph, num_comms))


def len_most_dense(graph, num_comms):
    """len_most_dense (the densest fluid community is the smallest)"""
    return len(eval_asyn(graph, num_comms)[-1])


def len_least_dense(graph, num_comms):
    """len_least_dense (the least dense fluid community is the largest)"""
    return len(eval_asyn(graph, num_comms)[0])


def partitions(graph, num_comms):
    """Partition of the whole graph: the fluid communities, plus each node outside the
    largest connected component as a singleton community."""
    communities = list(eval_asyn(graph, num_comms))
    covered = set().union(*communities)
    communities += [{n} for n in graph if n not in covered]
    return communities


class CommunitiesAsyn(FeatureClass):
    """Communities Asyn class.

    The asynchronous fluid communities algorithm is described in
    [1]_. The algorithm is based on the simple idea of fluids interacting
    in an environment, expanding and pushing each other. Its initialization is
    random, so we fix the seed to make the features reproducible.

    The algorithm needs a connected graph: it is run on the largest (weakly) connected
    component, treated as undirected. For the partition-quality features, the remaining nodes
    are added as singleton communities.
    The density of a community is the fluid density 1/|C|.

    References
    ----------
    .. [1] Parés F., Garcia-Gasulla D. et al. "Fluid Communities: A
       Competitive and Highly Scalable Community Detection Algorithm".
       [https://arxiv.org/pdf/1703.09307.pdf].

    """

    modes = ["slow"]
    shortname = "CA"
    name = "communities_asyn"
    encoding = "networkx"

    def compute_features(self):
        num_communities = [2, 4, 6, 8, 10]
        for num_comms in num_communities:
            self.add_feature(
                f"sum_density_c={num_comms}",
                partial(sum_density, num_comms=num_comms),
                f"The total density of communities after fluid optimisations for c={num_comms}",
                InterpretabilityScore(3),
            )

            self.add_feature(
                f"ratio_density_c={num_comms}",
                partial(ratio_density, num_comms=num_comms),
                f"The ratio density of communities after fluid optimisations for c={num_comms}",
                InterpretabilityScore(3),
            )
            self.add_feature(
                f"len_most_dense_c={num_comms}",
                partial(len_most_dense, num_comms=num_comms),
                f"The length of the densest community after fluid optimisations for c={num_comms}",
                InterpretabilityScore(4),
            )

            self.add_feature(
                f"len_least_dense_c={num_comms}",
                partial(len_least_dense, num_comms=num_comms),
                f"The length of the least dense community after fluid opt for c={num_comms}",
                InterpretabilityScore(4),
            )

            # computing clustering quality
            self.add_feature(
                f"partition_c={num_comms}",
                partial(partitions, num_comms=num_comms),
                f"The optimal partition after fluid optimisations for c={num_comms}",
                InterpretabilityScore(4),
                statistics="clustering",
            )
