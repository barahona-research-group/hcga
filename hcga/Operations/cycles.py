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

class Cycles():
    """
    Cycles class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}
        
    def feature_extraction(self):
        
        """Compute some simple cycle features of the network


        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        feature_list : dict
           Dictionary of features related to cycles.
           
           
        Notes
        -----
        Cycles calculations using networkx:
            `Networkx_cycles <https://networkx.github.io/documentation/stable/reference/algorithms/cycles.html>`_
        """
        

        G=self.G
        feature_list={}
        if not nx.is_directed(G) and nx.cycle_basis(G):
            # Find list of cycles for graph
            cycles=nx.cycle_basis(G)
            # Add basic cycle features 
            feature_list['num_cycles']=len(cycles)
            l=[len(i) for i in cycles]
            feature_list['mean_cycle_length']=np.mean(l)
            feature_list['shortest_cycle']=min(l)
            feature_list['longest_cycle']=max(l)
        else:
            feature_list['num_cycles']=np.nan
            feature_list['mean_cycle_length']=np.nan
            feature_list['shortest_cycle']=np.nan
            feature_list['longest_cycle']=np.nan
            

        self.features = feature_list
        
