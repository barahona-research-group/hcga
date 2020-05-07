"""Chemical theory class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "ChemicalTheory"


class ChemicalTheory(FeatureClass):
    """Chemical theory class."""

    modes = ["fast", "medium", "slow"]
    shortname = "CT"
    name = "chemical_theory"
    encoding = "networkx"

    def compute_features(self):
        """Compute some chemical theory measures for the network.

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names
        """

        wiener_index = lambda graph: nx.wiener_index(graph)
        self.add_feature(
            "wiener index",
            wiener_index,
            "The wiener index is defined as the sum of the lengths of the shortest paths between all pairs of vertices",
            InterpretabilityScore(4),
        )

        estrada_index = lambda graph: nx.estrada_index(graph)
        self.add_feature(
            "estrada_index",
            estrada_index,
            "The Estrada Index is a topological index of protein folding or 3D compactness",
            InterpretabilityScore(4),
        )
