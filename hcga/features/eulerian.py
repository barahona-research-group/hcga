"""Eulerian Measures class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Eulerian"


class Eulerian(FeatureClass):
    """Eulerian Measures class.

    Features based on Eulerian circuits. An *Eulerian circuit* is a closed walk
    that includes each edge of a graph exactly once.

    Uses networkx, see 'https://networkx.org/documentation/stable/reference/algorithms/euler.html`

    References
    ----------
    .. [1] J. Edmonds, E. L. Johnson.
       Matching, Euler tours and the Chinese postman.
       Mathematical programming, Volume 5, Issue 1 (1973), 111-114.
    .. [2] https://en.wikipedia.org/wiki/Eulerian_path

    """

    modes = ["fast", "medium", "slow"]
    shortname = "EU"
    name = "eulerian"
    encoding = "networkx"

    def compute_features(self):
        # checking if eulerian
        self.add_feature(
            "eulerian",
            nx.is_eulerian,
            "A graph is eulerian if it has a eulerian circuit: a closed walk that includes \
            each edges of the graph exactly once",
            InterpretabilityScore(3),
        )

        # checking if semi eulerian
        self.add_feature(
            "semi_eulerian",
            nx.is_semieulerian,
            "A graph is semi eulerian if it has a eulerian path but no eulerian circuit",
            InterpretabilityScore(3),
        )

        # checking if eulerian path exists
        self.add_feature(
            "semi_eulerian",
            nx.has_eulerian_path,
            "Whether a eulerian path exists in the network",
            InterpretabilityScore(3),
        )
