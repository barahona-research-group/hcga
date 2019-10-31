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


class IndependentSets():
    """
    Independent sets class
    """
    
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute independent set measures.

        An independent set is a set of nodes such that the subgraph
        of G induced by these nodes contains no edges. A maximal
        independent set is an independent set such that it is not possible
        to add a new node and still get an independent set.

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list :list
           List of features related to independent sets.


        Notes
        -----
        Eccentricity using networkx:
            `Networkx_maximal_indpendent_sets <https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.mis.maximal_independent_set.html#networkx.algorithms.mis.maximal_independent_set>`_   
        


        """
        

        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):            
    
    
            ind_set = nx.maximal_independent_set(G, seed = 10)
            
            feature_list['num_ind_nodes_norm']=len(ind_set)
            
            feature_list['ratio__ind_nodes_norm']=len(ind_set)/len(G)
        else:
            feature_list['num_ind_nodes_norm']=np.nan
            feature_list['ratio__ind_nodes_norm']=np.nan

        self.features = feature_list
