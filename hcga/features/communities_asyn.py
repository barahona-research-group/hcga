# -*- coding: utf-8 -*-
# This file is part of hcga.
#
# Copyright (C) 2019,
# Robert Peach (r.peach13@imperial.ac.uk),
# Alexis Arnaudon (alexis.arnaudon@epfl.ch),
# https://github.com/ImperialCollegeLondon/hcga.git
#
# hcga is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# hcga is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with hcga.  If not, see <http://www.gnu.org/licenses/>.

from collections import Counter
from functools import lru_cache

import numpy as np
from networkx.algorithms.components import is_connected
from networkx.exception import NetworkXError
from networkx.utils import groups, py_random_state

from . import utils
from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "CommunitiesAsyn"


class CommunitiesAsyn(FeatureClass):
    """
    Communities Asyn class
    """

    modes = ["medium", "slow"]
    shortname = "CA"
    name = "communities_asyn"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """Compute the measures about community detection using async fluid algorithm.

        Notes
        -----
        Currently hardcoded the number of communities tried:  np.linspace(2, 20, 10)
        """

        @lru_cache(maxsize=None)
        def eval_asyn(graph, num_comms):
            """this evaluates the main function and cach it for speed up"""
            graph = utils.ensure_connected(graph)

            # To not crash with trivial graph
            if int(num_comms) > len(graph):
                return [{0}], [0]

            return asyn_fluidc(graph, int(num_comms))

        num_communities = np.linspace(2, 20, 10)
        for num_comms in num_communities:
            self.add_feature(
                "sum_density_c={}".format(num_comms),
                lambda graph: sum(eval_asyn(graph, num_comms)[1]),
                "The total density of communities after async fluid optimisations for c={}".format(
                    num_comms
                ),
                InterpretabilityScore(3),
            )

            self.add_feature(
                "ratio_density_c={}".format(num_comms),
                lambda graph: np.min(eval_asyn(graph, num_comms)[1])
                / np.max(eval_asyn(graph, num_comms)[1]),
                "The ratio density of communities after async fluid optimisations for c={}".format(
                    num_comms
                ),
                InterpretabilityScore(3),
            )

            self.add_feature(
                "len_most_dense_c={}".format(num_comms),
                lambda graph: len(
                    eval_asyn(graph, num_comms)[0][
                        np.argmax(eval_asyn(graph, num_comms)[1])
                    ]
                ),
                "The length of the most dense community after async fluid optimisations for c={}".format(
                    num_comms
                ),
                InterpretabilityScore(4),
            )

            self.add_feature(
                "len_least_dense_c={}".format(num_comms),
                lambda graph: len(
                    eval_asyn(graph, num_comms)[0][
                        np.argmin(eval_asyn(graph, num_comms)[1])
                    ]
                ),
                "The length of the least dense community after async fluid optimisations for c={}".format(
                    num_comms
                ),
                InterpretabilityScore(4),
            )

            # computing clustering quality
            self.add_feature(
                "partition_c={}".format(num_comms),
                lambda graph: eval_asyn(graph, num_comms)[0],
                "The optimal partition after async fluid optimisations for c={}".format(
                    num_comms
                ),
                InterpretabilityScore(4),
            )


# this function is adapted from networks directly
@py_random_state(3)
def asyn_fluidc(G, k, max_iter=100, seed=None):
    """Returns communities in `G` as detected by Fluid Communities algorithm.

    The asynchronous fluid communities algorithm is described in
    [1]_. The algorithm is based on the simple idea of fluids interacting
    in an environment, expanding and pushing each other. It's initialization is
    random, so found communities may vary on different executions.

    The algorithm proceeds as follows. First each of the initial k communities
    is initialized in a random vertex in the graph. Then the algorithm iterates
    over all vertices in a random order, updating the community of each vertex
    based on its own community and the communities of its neighbours. This
    process is performed several times until convergence.
    At all times, each community has a total density of 1, which is equally
    distributed among the vertices it contains. If a vertex changes of
    community, vertex densities of affected communities are adjusted
    immediately. When a complete iteration over all vertices is done, such that
    no vertex changes the community it belongs to, the algorithm has converged
    and returns.

    This is the original version of the algorithm described in [1]_.
    Unfortunately, it does not support weighted graphs yet.

    Parameters
    ----------
    G : Graph

    k : integer
        The number of communities to be found.

    max_iter : integer
        The number of maximum iterations allowed. By default 15.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    communities : iterable
        Iterable of communities given as sets of nodes.

    Notes
    -----
    k variable is not an optional argument.

    References
    ----------
    .. [1] Parés F., Garcia-Gasulla D. et al. "Fluid Communities: A
       Competitive and Highly Scalable Community Detection Algorithm".
       [https://arxiv.org/pdf/1703.09307.pdf].
    """
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
            if len(com_counter.keys()) > 0:
                max_freq = max(com_counter.values())
                best_communities = [
                    com
                    for com, freq in com_counter.items()
                    if (max_freq - freq) < 0.0001
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
