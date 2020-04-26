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

import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore
from .utils import ensure_connected

featureclass_name = "Hits"


class Hits(FeatureClass):
    """
    HITS hubs class
    """

    modes = ["medium", "slow"]
    shortname = "LAH"
    name = "Hits"
    encoding = "networkx"

    def compute_features(self):
        """
        Computes measures based on the HITS hubs 
        
        The HITS algorithm computes two numbers for a node [1]_ [2]_.
        Authorities estimates the node value based on the incoming links.
        Hubs estimates the node value based on outgoing links.


        Notes
        -----
                Hits calculations using networkx:
            `Networkx_hits <https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.hits_alg.hits.html#networkx.algorithms.link_analysis.hits_alg.hits>`_
        
 
        The eigenvector calculation is done by the power iteration method
        and has no guarantee of convergence.  The iteration will stop
        after max_iter iterations or an error tolerance of
        number_of_nodes(G)*tol has been reached.
    
        The HITS algorithm was designed for directed graphs but this
        algorithm does not check if the input graph is directed and will
        execute on undirected graphs.
    
        References
        ----------
        .. [1] A. Langville and C. Meyer,
           "A survey of eigenvector methods of web information retrieval."
           http://citeseer.ist.psu.edu/713792.html
        .. [2] Jon Kleinberg,
           Authoritative sources in a hyperlinked environment
           Journal of the ACM 46 (5): 604-32, 1999.
           doi:10.1145/324133.324140.
           http://www.cs.cornell.edu/home/kleinber/auth.pdf.

        """

        def hits(graph):
            h, a = nx.hits_numpy(ensure_connected(graph))
            return list(h.values())

        self.add_feature(
            "Hits",
            hits,
            "Hits algorithm",
            InterpretabilityScore(3),
            statistics="centrality",
        )
