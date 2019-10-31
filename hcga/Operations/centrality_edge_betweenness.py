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

from networkx.algorithms import centrality
from hcga.Operations import utils
import numpy as np



class EdgeBetweennessCentrality():
    """
    Edge betweenness centrality class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):
        """
        Compute the edge betweenness centrality for nodes.
        
        The edge betweeness centrality for an edge is the number of shortest
        paths that pass through it.
        Here the fraction of shortest paths that pass through
        each edge is calculated.
        
        Parameters
        ----------
        G : graph
          A networkx graph


        Returns
        -------
        feature_list :list
           List of features related to betweenness centrality.
           
           
        Notes
        -----
        Edge Betweenness centrality calculations using networkx:
            `Networkx_centrality <https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.edge_betweenness_centrality.html>`_
        """     
                
        # Defining the input arguments
        bins = [10,20,50]
        

        G = self.G

        feature_list = {}

        #Calculate the edge betweenness centrality of each node
        edge_betweenness_centrality = np.asarray(list(centrality.edge_betweenness_centrality(G).values()))

        # Basic stats regarding the edge betweenness centrality distribution
        feature_list['mean'] = edge_betweenness_centrality.mean()
        feature_list['std'] = edge_betweenness_centrality.std()
        feature_list['max'] = edge_betweenness_centrality.max()
        feature_list['min'] = edge_betweenness_centrality.min()
        
        for i in range(len(bins)):

            
            # Fitting the edge betweenness centrality distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(edge_betweenness_centrality,bins=bins[i])
            feature_list['opt_model_{}'.format(bins[i])] = opt_mod

            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(edge_betweenness_centrality,bins=bins[i])[0][-2]# value 'a' in power law
            feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(edge_betweenness_centrality,bins=bins[i])[1] # value sse in power law



        
        self.features = feature_list
