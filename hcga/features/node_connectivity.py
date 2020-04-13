# -*- coding: utf-8 -*-
# This file is part of hcga.
#
# Copyright (C) 2019, 
# Robert Peach (r.peach13@imperial.ac.uk), 
# Alexis Arnaudon (alexis.arnaudon@epfl.ch), 
# https://github.com/ImperialCollegeLondon/hcga.git
#
# hcga is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# hcga is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with hcga.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "NodeConnectivity"

class NodeConnectivity(FeatureClass):
    modes = ["slow"]
    shortname = "NC"
    name = "node_connectivity"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """Compute node connectivity measures.

        Node connectivity is equal to the minimum number of nodes that
            must be removed to disconnect G or render it trivial. If source
            and target nodes are provided, this function returns the local node
            connectivity: the minimum number of nodes that must be removed to break
            all paths from source to target in G.


        Notes
        -----
        Implementation of networkx code:
            `Networkx_node_connectivity <https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/connectivity/connectivity.html#node_connectivity>`_

        This is a flow based implementation of node connectivity. The
        algorithm works by solving $O((n-\delta-1+\delta(\delta-1)/2))$
        maximum flow problems on an auxiliary digraph. Where $\delta$
        is the minimum degree of G. For details about the auxiliary
        digraph and the computation of local node connectivity see
        :meth:`local_node_connectivity`. This implementation is based
        on algorithm 11 in [1]_.
        
        References
        ----------
        .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for 
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf

        """
        

        def node_conn(graph):
            # calculating node connectivity
            node_connectivity = nx.all_pairs_node_connectivity(graph)
            N = graph.number_of_nodes()
            
            node_conn = np.zeros([N,N])
            for key1, value1 in node_connectivity.items():    
                for key2, value2 in value1.items():
                    node_conn[key1,key2] = value2
            return list(np.triu(node_conn).flatten())

        self.add_feature(
            "node_conn",
            node_conn,
            "Node connectivity (statistics)",
            InterpretabilityScore("max")-1,
            statistics="centrality",
        )

        
        # Calculate connectivity
        self.add_feature(
            "node_connectivity",
            nx.node_connectivity,
            "Node connectivity",
            InterpretabilityScore("max")-1,
        )
        self.add_feature(
            "avg_node_connectivity",
            nx.average_node_connectivity,
            "Average node connectivity",
            InterpretabilityScore("max")-1,
        )
        self.add_feature(
            "edge_connectivity",
            nx.edge_connectivity,
            "Edge connectivity",
            InterpretabilityScore("max")-1,
        )
        
        # calculate the wiener index 
        self.add_feature(
            "wiener_index",
            nx.wiener_index,
            "Wiener index",
            InterpretabilityScore("max")-1,
        )
        

