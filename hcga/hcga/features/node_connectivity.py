"""Node connectivity class."""
import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "NodeConnectivity"


class NodeConnectivity(FeatureClass):
    """Node connectivity class.

    Implementation of networkx code:
            `Networkx_node_connectivity <https://networkx.github.io/documentation/stable/_modules/\
            networkx/algorithms/connectivity/connectivity.html#node_connectivity>`_
    """

    modes = ["slow"]
    shortname = "NC"
    name = "node_connectivity"
    encoding = "networkx"

    def compute_features(self):
        def node_conn(graph):
            # calculating node connectivity
            node_connectivity = nx.all_pairs_node_connectivity(graph)
            N = graph.number_of_nodes()

            node_conn = np.zeros([N, N])
            for key1, value1 in node_connectivity.items():
                for key2, value2 in value1.items():
                    node_conn[key1, key2] = value2
            return list(np.triu(node_conn).flatten())

        self.add_feature(
            "node_conn",
            node_conn,
            "Node connectivity (statistics)",
            InterpretabilityScore("max") - 1,
            statistics="centrality",
        )

        # Calculate connectivity
        self.add_feature(
            "node_connectivity",
            nx.node_connectivity,
            "Node connectivity",
            InterpretabilityScore("max") - 1,
        )
        self.add_feature(
            "avg_node_connectivity",
            nx.average_node_connectivity,
            "Average node connectivity",
            InterpretabilityScore("max") - 1,
        )
        self.add_feature(
            "edge_connectivity",
            nx.edge_connectivity,
            "Edge connectivity",
            InterpretabilityScore("max") - 1,
        )

        # calculate the wiener index


#        self.add_feature(
#            "wiener_index",
#            nx.wiener_index,
#            "Wiener index",
#            InterpretabilityScore("max") - 1,
#        )
