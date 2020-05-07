"""Small worldness class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "SmallWorldness"


class SmallWorldness(FeatureClass):
    """Small worldness class."""

    modes = ["slow"]
    shortname = "SW"
    name = "small_worldness"
    encoding = "networkx"

    def compute_features(self):
        """Compute the small world measures of the network.

        The small-world coefficient of a graph G is:

        omega = Lr/L - C/Cl

        where C and L are respectively the average clustering coefficient and
        average shortest path length of G. Lr is the average shortest path length
        of an equivalent random graph and Cl is the average clustering coefficient
        of an equivalent lattice graph.

        The small-world coefficient (omega) ranges between -1 and 1. Values close
        to 0 means the G features small-world characteristics. Values close to -1
        means G has a lattice shape whereas values close to 1 means G is a random
        graph.

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------

        Notes
        -----
        Small world calculations using networkx:
            `Networkx_omega <https://networkx.github.io/documentation/latest/\
            _modules/networkx/algorithms/smallworld.html#omega>`_

        The implementation is adapted from the algorithm by Telesford et al. [1]_.

        References
        ----------
        .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
               "The Ubiquity of Small-World Networks".
               Brain Connectivity. 1 (0038): 367-75.
        """

        # omega metric
        self.add_feature(
            "omega",
            lambda graph: nx.omega(graph),
            "The small world coefficient omega",
            InterpretabilityScore(4),
        )
