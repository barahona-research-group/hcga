"""Chemical theory class."""
from functools import partial

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "ChemicalTheory"


class ChemicalTheory(FeatureClass):
    """Chemical theory class."""

    modes = ["fast", "medium", "slow"]
    shortname = "CT"
    name = "chemical_theory"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "wiener index",
            nx.wiener_index,
            "Sum of the lengths of the shortest paths between all pairs of vertices",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "wiener index weighted",
            partial(nx.wiener_index, weight="weight"),
            "Sum of the lengths of the shortest paths between all pairs of vertices",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "estrada_index",
            nx.estrada_index,
            "Topological index of protein folding or 3D compactness",
            InterpretabilityScore(4),
        )
