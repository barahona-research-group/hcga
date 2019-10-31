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
from hcga.Operations import utils
import numpy as np
import networkx as nx

class NodeCliqueNumber():
    """
    Node clique number class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):
        """
        Compute the maximal clique containing each node, i.e passing through
        that node.
        
        
        Parameters
        ----------
        G : graph
          A networkx graph


        Returns
        -------
        feature_list : dict
           Dictionary of features related to clique number.


        Notes
        -----
        Clique number calculations using networkx:
            `Networkx_clique <https://networkx.github.io/documentation/stable/reference/algorithms/clique.html>`_
        """        
                
        # Defining the input arguments
        bins = [10,20,50]
        


        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):
            #Calculate the largest maximal clique containing each node
            node_clique_number = np.asarray(list(clique.node_clique_number(G).values()))
    
            # Basic stats regarding the node clique number distribution
            feature_list['mean'] = node_clique_number.mean()
            feature_list['std'] = node_clique_number.std()
            feature_list['max'] = node_clique_number.max()
            feature_list['min'] = node_clique_number.min()
            
            for i in range(len(bins)):

                
                # Fitting the node clique number distribution and finding the optimal
                # distribution according to SSE
                opt_mod,opt_mod_sse = utils.best_fit_distribution(node_clique_number,bins=bins[i])
                feature_list['opt_model_{}'.format(bins[i])] = opt_mod
    
                # Fitting power law and finding 'a' and the SSE of fit.
                feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(node_clique_number,bins=bins[i])[0][-2]# value 'a' in power law
                feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(node_clique_number,bins=bins[i])[1] # value sse in power law
        else:
            feature_list['mean'] = np.nan
            feature_list['std'] = np.nan
            feature_list['max'] = np.nan
            feature_list['min'] = np.nan
            for i in range(len(bins)):
                feature_list['opt_model_{}'.format(bins[i])] = np.nan
                feature_list['powerlaw_a_{}'.format(bins[i])] = np.nan
                feature_list['powerlaw_SSE_{}'.format(bins[i])] = np.nan


        self.features = feature_list
