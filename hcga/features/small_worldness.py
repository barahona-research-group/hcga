"""Small worldness class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "SmallWorldness"


class SmallWorldness(FeatureClass):
    """Small worldness class.

    Fetures based on small-worldness, where the small-world coefficient of a graph G is:

    omega = Lr/L - C/Cl

    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Lr is the average shortest path length
    of an equivalent random graph and Cl is the average clustering coefficient
    of an equivalent lattice graph.

    Small world calculations using networkx:
        `Networkx_omega <https://networkx.github.io/documentation/latest/\
        _modules/networkx/algorithms/smallworld.html#omega>`_

    References
    ----------
    .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
           "The Ubiquity of Small-World Networks".
           Brain Connectivity. 1 (0038): 367-75.  PMC 3604768. PMID 22432451.
           doi:10.1089/brain.2011.0038.

    """

    modes = ["slow"]
    shortname = "SW"
    name = "small_worldness"
    encoding = "networkx"

    def compute_features(self):
        # omega metric
        self.add_feature(
            "omega",
            nx.omega,
            "The small world coefficient omega",
            InterpretabilityScore(4),
        )
