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
from hcga.Operations import utils
import numpy as np
from hcga.Operations.utils import summary_statistics

class CoreNumber():
    """
    Core number class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}
        
    def feature_extraction(self):
        """
        Compute core number for each node
        
        A k-core is the maximal subgraph that contains nodes of degree k or 
        higher. 
        The core number for a node is the largest value of k such that a 
        k-core containing that node exists [1]_.
        
                Parameters
        ----------
        G : graph
          A networkx graph


        Returns
        -------
        feature_list : dict
           Dictionary of features related to core number.


        Notes
        -----
        Core number calculations using networkx:
            `Networkx_core_number <https://networkx.github.io/documentation/stable/reference/algorithms/core.html>`_
            
        References
        ----------
        .. [1] An O(m) Algorithm for Cores Decomposition of Networks
           Vladimir Batagelj and Matjaz Zaversnik,  2003.
           http://arxiv.org/abs/cs.DS/0310049            
            
        """
        
        # Defining the input arguments
        bins = [10,20,50]
        

        G = self.G
        feature_list = {}
        
        
        G.remove_edges_from(nx.selfloop_edges(G))
        
        #Calculate the core number of each node
        core_number = np.asarray(list(nx.core_number(G).values()))
        
        # Basic stats regarding the core number distribution     
        
        feature_list = summary_statistics(feature_list,core_number,'')       

            
        for i in range(len(bins)):

            # Fitting the core number and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(core_number,bins=bins[i])
            feature_list['opt_model_{}'.format(bins[i])] = opt_mod
    
            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(core_number,bins=bins[i])[0][-2]# value 'a' in power law
            feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(core_number,bins=bins[i])[1] # value sse in power law

        

        self.features = feature_list
