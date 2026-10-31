"""Independent sets class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "IndependentSets"

SEED = 0


def size_max_indep_set(graph):
    """Size of a random maximal independent set (seeded; directed graphs treated as undirected)."""
    if graph.is_directed():
        graph = graph.to_undirected(as_view=True)
    return len(nx.maximal_independent_set(graph, seed=SEED))


class IndependentSets(FeatureClass):
    """Independent sets class.

    Features based on independent sets.

    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges. A maximal
    independent set is an independent set such that it is not possible
    to add a new node and still get an independent set.

    Uses networkx, see 'https://networkx.org/documentation/stable/reference/algorithms/mis.html`

    """

    modes = ["fast", "medium", "slow"]
    shortname = "IS"
    name = "independent_sets"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "size_max_indep_set",
            size_max_indep_set,
            "The number of nodes in the maximal independent set",
            InterpretabilityScore(3),
        )
