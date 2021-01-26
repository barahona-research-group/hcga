"""Structural Holes class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "StructuralHoles"


def constraint(graph):
    return list(nx.structuralholes.constraint(graph).values())


def effective_size(graph):
    return list(nx.structuralholes.effective_size(graph).values())


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
            constraint,
            "The constraint is a measure of the extent to which a node v is invested in \
            those nodes that are themselves invested in the neighbors of v",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "effective_size",
            effective_size,
            "The effective size of a node’s ego network is based on the concept of redundancy. \
            A person’s ego network has redundancy to the extent that her contacts are connected \
            to each other as well. ",
            InterpretabilityScore(3),
            statistics="centrality",
        )
