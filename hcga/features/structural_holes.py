"""Structural Holes class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "StructuralHoles"


class StructuralHoles(FeatureClass):
    """Structural Holes class."""

    modes = ["medium", "slow"]
    shortname = "SH"
    name = "structural_holes"
    encoding = "networkx"

    def compute_features(self):

        # distribution of structural holes constraint
        self.add_feature(
            "constraint",
            lambda graph: list(nx.structuralholes.constraint(graph).values()),
            "The constraint is a measure of the extent to which a node v is invested in \
            those nodes that are themselves invested in the neighbors of v",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "effective_size",
            lambda graph: list(nx.structuralholes.effective_size(graph).values()),
            "The effective size of a node’s ego network is based on the concept of redundancy. \
            A person’s ego network has redundancy to the extent that her contacts are connected \
            to each other as well. ",
            InterpretabilityScore(3),
            statistics="centrality",
        )
