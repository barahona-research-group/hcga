"""Core number class."""
import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import remove_selfloops

featureclass_name = "CoreNumber"


def core_number(graph):
    """core_number"""
    return list(np.asarray(list(nx.core_number(remove_selfloops(graph)).values())))


class CoreNumber(FeatureClass):
    """Core number class."""

    modes = ["fast", "medium", "slow"]
    shortname = "CoN"
    name = "core_number"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "core number",
            core_number,
            "The core number distribution",
            InterpretabilityScore(5),
            statistics="centrality",
        )
