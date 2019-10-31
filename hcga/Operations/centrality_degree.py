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



class DegreeCentrality():
    """
    Degree centrality class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):
        """
        Compute the degree centrality for nodes.

        The degree centrality for a node v is the fraction of nodes it
        is connected to.


        Parameters
        ----------
        G : graph
          A networkx graph


        Returns
        -------
        feature_list :list
           List of features related to degree centrality.


        Notes
        -----
        Degree centrality calculations using networkx:
            `Networkx_centrality <https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/centrality/degree_alg.html#degree_centrality>`_
            
        The degree centrality values are normalized by dividing by the maximum
        possible degree in a simple graph n-1 where n is the number of nodes 
        in G.

        For multigraphs or graphs with self loops the maximum degree might
        be higher than n-1 and values of degree centrality greater than 1
        are possible.
        """
        # Defining the input arguments
        bins = [10,20,50]
        


        G = self.G

        feature_list = {}

        #Calculate the degree centrality of each node
        degree_centrality = np.asarray(list(centrality.degree_centrality(G).values()))

        # Basic stats regarding the degree centrality distribution
        feature_list['mean'] = degree_centrality.mean()
        feature_list['std'] = degree_centrality.std()
        feature_list['max'] = degree_centrality.max()
        feature_list['min'] = degree_centrality.min()
        
        for i in range(len(bins)):

            # Fitting the degree centrality distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(degree_centrality,bins=bins[i])
            feature_list['opt_model_{}'.format(bins[i])] = opt_mod

            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(degree_centrality,bins=bins[i])[0][-2]# value 'a' in power law
            feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(degree_centrality,bins=bins[i])[1] # value sse in power law



        self.features = feature_list
