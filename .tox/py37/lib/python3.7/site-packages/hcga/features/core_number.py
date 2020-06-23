"""Core number class."""
import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore
from .utils import remove_selfloops

featureclass_name = "CoreNumber"


class CoreNumber(FeatureClass):
    """Core number class."""

    modes = ["fast", "medium", "slow"]
    shortname = "CoN"
    name = "core_number"
    encoding = "networkx"

    def compute_features(self):

        core_number = lambda graph: list(
            np.asarray(list(nx.core_number(remove_selfloops(graph)).values()))
        )
        self.add_feature(
            "core number",
            core_number,
            "The core number distribution",
            InterpretabilityScore(5),
            statistics="centrality",
        )
