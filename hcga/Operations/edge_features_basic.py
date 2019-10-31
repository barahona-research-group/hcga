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

from hcga.Operations.utils import summary_statistics

class EdgeFeaturesBasic():
    """
    Edge features class
    """    
    
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}
        
    def feature_extraction(self):
        """
        Compute basic statistics of the edge features.
        
        The edge features (usually just edge weights) provide an additional dimension to describe our graph/
        
        Parameters
        ----------
        G : graph
          A networkx graph


        Returns
        -------
        feature_list : dict
           Dictionary of features related to edge features.


        Notes
        -----
       
            
        

        
        """        
        G = self.G
        
        
        feature_list = {}
        

        edge_attributes = list(nx.get_edge_attributes(G,'weight').values())
        feature_list = summary_statistics(feature_list,edge_attributes,'edge_weights')       
        
        
        self.features=feature_list
