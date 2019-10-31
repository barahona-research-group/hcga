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

class Diameter():
    """
    Diameter class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):
        """
        Compute diameter and radius of graph
        
        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        feature_list : dict
           dictionary containing diameter and radius of graph.
           
        Notes
        -----
        Diameter/radius calculations using networkx:
            `Networkx_diameter/radius <https://networkx.github.io/documentation/stable/reference/algorithms/distance_measures.html>`_
        """
        

        
        G = self.G
        feature_list = {}
        if not nx.is_directed(G) or (nx.is_directed(G) and nx.is_strongly_connected(G)):
            #Adding diameter and radius 
            feature_list['diameter']=nx.diameter(G)
            feature_list['radius']=nx.radius(G)
        else:
            feature_list['diameter']=np.nan
            feature_list['radius']=np.nan

        self.features = feature_list
        
        
