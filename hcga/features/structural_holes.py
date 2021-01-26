"""Structural Holes class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "StructuralHoles"


class StructuralHoles(FeatureClass):
    """Structural Holes class.


    Holes calculations using networkx:
        `Structural Holes <https://networkx.org/documentation/stable//reference/algorithms/structuralholes.html>`_

    The *constraint* is a measure of the extent to which a node *v* is
    invested in those nodes that are themselves invested in the
    neighbors of *v* [1]_.

    The *effective size* of a node's ego network is based on the concept
    of redundancy. A person's ego network has redundancy to the extent
    that her contacts are connected to each other as well. The
    nonredundant part of a person's relationships it's the effective
    size of her ego network [2]_.

    References
    ----------
    .. [1] Burt, Ronald S.
           "Structural holes and good ideas".
           American Journal of Sociology (110): 349–399.
    .. [2] Burt, Ronald S.
           *Structural Holes: The Social Structure of Competition.*
           Cambridge: Harvard University Press, 1995.
    .. [3] Borgatti, S.
           "Structural Holes: Unpacking Burt's Redundancy Measures"
           CONNECTIONS 20(1):35-38.
           http://www.analytictech.com/connections/v20(1)/holes.htm

    """

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
