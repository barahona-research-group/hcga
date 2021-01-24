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

        wiener_index = lambda graph: nx.wiener_index(graph)
        self.add_feature(
            "wiener index",
            wiener_index,
            "Sum of the lengths of the shortest paths between all pairs of vertices",
            InterpretabilityScore(4),
        )

        wiener_index = lambda graph: nx.wiener_index(graph, weight="weight")
        self.add_feature(
            "wiener index weighted",
            wiener_index,
            "Sum of the lengths of the shortest paths between all pairs of vertices",
            InterpretabilityScore(4),
        )

        estrada_index = lambda graph: nx.estrada_index(graph)
        self.add_feature(
            "estrada_index",
            estrada_index,
            "Topological index of protein folding or 3D compactness",
            InterpretabilityScore(4),
        )
