"""Chemical theory class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "ChemicalTheory"


class ChemicalTheory(FeatureClass):
    """Chemical theory class.

    Here we implement the wiener and estrada indexes.

    The *Wiener index* of a graph is the sum of the shortest-path
    distances between each pair of reachable nodes [1]_. For pairs of nodes
    in undirected graphs, only one orientation of the pair is counted.

    The Estrada Index is a topological index of folding or 3D “compactness” [2]_.

    References
    ----------
    .. [1] Rouvray, Dennis H.
     "The rich legacy of half a century of the Wiener index.",
      Topology in Chemistry. Woodhead Publishing, 2002. 16-37.
    .. [2] E. Estrada,  Characterization of 3D molecular structure,
       Chem. Phys. Lett. 319, 713 (2000).

    """

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
