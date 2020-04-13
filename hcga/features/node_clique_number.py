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

from networkx.algorithms import clique
from . import utils
from ..feature_class import FeatureClass, InterpretabilityScore
import numpy as np
import networkx as nx

featureclass_name = "NodeCliqueNumber"

class NodeCliqueNumber(FeatureClass):
    """
    Node clique number class
    """
 
    modes = ["fast", "medium", "slow"]
    shortname = "CN"
    name = "node_clique_number"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute the maximal clique containing each node, i.e passing through
        that node.
        
        
        Notes
        -----
        Clique number calculations using networkx:
            `Networkx_clique <https://networkx.github.io/documentation/stable/reference/algorithms/clique.html>`_
        """        
                
        
        self.add_feature(
            "clique sizes",
            lambda graph: np.asarray(list(clique.node_clique_number(utils.ensure_connected(graph)).values())),
            "the distribution of clique sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )
