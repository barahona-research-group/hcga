"""Communities Asyn class."""

from collections import Counter
from functools import lru_cache, partial

import numpy as np
from networkx.algorithms.components import is_connected
from networkx.exception import NetworkXError
from networkx.utils import groups, py_random_state

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "CommunitiesAsyn"


@lru_cache(maxsize=None)
def eval_asyn(graph, num_comms):
    """this evaluates the main function and cach it for speed up."""
    return asyn_fluidc(ensure_connected(graph), int(num_comms))


def sum_density(graph, num_comms):
    """sum_density"""
    return sum(eval_asyn(graph, num_comms)[1])


def ratio_density(graph, num_comms):
    """"""
    return np.min(eval_asyn(graph, num_comms)[1]) / np.max(eval_asyn(graph, num_comms)[1])


def len_most_dense(graph, num_comms):
    """len_most_dense"""
    return len(eval_asyn(graph, num_comms)[0][np.argmax(eval_asyn(graph, num_comms)[1])])


def len_least_dense(graph, num_comms):
    """len_least_dense"""
    return len(eval_asyn(graph, num_comms)[0][np.argmin(eval_asyn(graph, num_comms)[1])])


def partitions(graph, num_comms):
    """partitions"""
    return eval_asyn(graph, num_comms)[0]


class CommunitiesAsyn(FeatureClass):
    """Communities Asyn class.

    The asynchronous fluid communities algorithm is described in
    [1]_. The algorithm is based on the simple idea of fluids interacting
    in an environment, expanding and pushing each other. Its initialization is
    random, so found communities may vary on different executions.

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
        num_communities = np.linspace(2, 10, 5)
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
                partial(len_most_dense, num_comms),
                f"The length of the densest community after fluid optimisations for c={num_comms}",
                InterpretabilityScore(4),
            )

            self.add_feature(
                f"len_least_dense_c={num_comms}",
                partial(len_least_dense, num_comms),
                f"The length of the least dense community after fluid opt for c={num_comms}",
                InterpretabilityScore(4),
            )

            # computing clustering quality
            self.add_feature(
                f"partition_c={num_comms}",
                partial(partitions, num_comms),
                f"The optimal partition after fluid optimisations for c={num_comms}",
                InterpretabilityScore(4),
                statistics="clustering",
            )


@py_random_state(3)
def asyn_fluidc(G, k, max_iter=100, seed=None):
    # noqa, pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """This function is adapted from networks directly."""
    # Initial checks
    if not isinstance(k, int):
        raise NetworkXError("k must be an integer.")
    if not k > 0:
        raise NetworkXError("k must be greater than 0.")
    if not is_connected(G):
        raise NetworkXError("Fluid Communities require connected Graphs.")
    if len(G) < k:
        raise NetworkXError("k cannot be bigger than the number of nodes.")
    # Initialization
    max_density = 1.0
    vertices = list(G)
    seed.shuffle(vertices)
    communities = {n: i for i, n in enumerate(vertices[:k])}
    density = {}
    com_to_numvertices = {}
    for vertex in communities.keys():
        com_to_numvertices[communities[vertex]] = 1
        density[communities[vertex]] = max_density
    # Set up control variables and start iterating
    iter_count = 0
    cont = True
    while cont:
        cont = False
        iter_count += 1
        # Loop over all vertices in graph in a random order
        vertices = list(G)
        seed.shuffle(vertices)
        for vertex in vertices:
            # Updating rule
            com_counter = Counter()
            # Take into account self vertex community
            try:
                com_counter.update({communities[vertex]: density[communities[vertex]]})
            except KeyError:
                pass
            # Gather neighbour vertex communities
            for v in G[vertex]:
                try:
                    com_counter.update({communities[v]: density[communities[v]]})
                except KeyError:
                    continue
            # Check which is the community with highest density
            new_com = -1
            if com_counter.keys():
                max_freq = max(com_counter.values())
                best_communities = [
                    com for com, freq in com_counter.items() if (max_freq - freq) < 0.0001
                ]
                # If actual vertex com in best communities, it is preserved
                try:
                    if communities[vertex] in best_communities:
                        new_com = communities[vertex]
                except KeyError:
                    pass
                # If vertex community changes...
                if new_com == -1:
                    # Set flag of non-convergence
                    cont = True
                    # Randomly chose a new community from candidates
                    new_com = seed.choice(best_communities)
                    # Update previous community status
                    try:
                        com_to_numvertices[communities[vertex]] -= 1
                        density[communities[vertex]] = (
                            max_density / com_to_numvertices[communities[vertex]]
                        )
                    except KeyError:
                        pass
                    # Update new community status
                    communities[vertex] = new_com
                    com_to_numvertices[communities[vertex]] += 1
                    density[communities[vertex]] = (
                        max_density / com_to_numvertices[communities[vertex]]
                    )
        # If maximum iterations reached --> output actual results
        if iter_count > max_iter:
            break
    # Return results by grouping communities as list of vertices
    return list(iter(groups(communities).values())), list(density.values())
