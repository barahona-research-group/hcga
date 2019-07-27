#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 20:24:06 2019

@author: Henry
"""

from networkx.algorithms import centrality
from hcga.Operations import utils
import numpy as np
import networkx as nx



class SecondOrderCentrality():
    """
    Second order centrality class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):
        """Compute the second order centrality for nodes.

        The second order centrality of a node is the standard deviation of 
        the return time to that node via a perpetual random walk.


        Parameters
        ----------
        G : graph
          A networkx graph

        
        Returns
        -------
        feature_list :list
           List of features related to second order centrality.


        Notes
        -----
        Second order centrality calculations using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html 
        """
        
        # Defining the input arguments
        bins = [10,20,50]
        
        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):
            # Calculate the second order centrality of each node
            second_order_centrality = np.asarray(list(centrality.second_order_centrality(G).values()))
        
            # Basic stats regarding the second order centrality distribution
            feature_list['mean'] = second_order_centrality.mean()
            feature_list['std'] = second_order_centrality.std()
            feature_list['max'] = second_order_centrality.max()
            feature_list['min'] = second_order_centrality.min()
                
                
            for i in range(len(bins)):
                """# Adding to feature names
                feature_names.append('opt_model_{}'.format(bins[i]))
                feature_names.append('powerlaw_a_{}'.format(bins[i]))
                feature_names.append('powerlaw_SSE_{}'.format(bins[i]))"""
                    
                # Fitting the second order centrality distribution and finding the optimal
                # distribution according to SSE
                opt_mod,opt_mod_sse = utils.best_fit_distribution(second_order_centrality,bins=bins[i])
                feature_list['opt_model_{}'.format(bins[i])] = opt_mod
        
                # Fitting power law and finding 'a' and the SSE of fit.
                feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(second_order_centrality,bins=bins[i])[0][-2]# value 'a' in power law
                feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(second_order_centrality,bins=bins[i])[1] # value sse in power law
        else:
            feature_list['second_order_centrality_features']= 'unavailable for directed graphs'


        self.features = feature_list