# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import numpy as np
import networkx as nx

class SmallWorld():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """ 
        Calculating metrics about small-world networks

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list :list
           List of features related to small worldness


        Notes
        -----


        """
        
        """
        self.feature_names = ['sigma','omega']
        """

        G = self.G

        feature_list = {}

        N = G.number_of_nodes()
        E = G.number_of_edges()

        
        #C = nx.transitivity(G)
        #L = nx.average_shortest_path_length(G)
        
        # sigma requires recalculating the average clustering effiient 
        # and the average shortest path length. These can be reused from
        # other features.

        #feature_list.append(nx.sigma(G))
        #feature_list.append(nx.omega(G))
        # small worldness sigma
#        nrand=20
#        
#        randMetrics = {"C": [], "L": []}
#        for i in range(nrand):
#            Gr = nx.dense_gnm_random_graph(N,E)
#            if nx.is_connected(Gr):
#                randMetrics["C"].append(nx.transitivity(Gr))
#                randMetrics["L"].append(nx.average_shortest_path_length(Gr))
#            
#        Cr = np.mean(randMetrics["C"])
#        Lr = np.mean(randMetrics["L"])
#        
#        sigma = (C / Cr) / (L / Lr)
        
        #feature_list['sigma']=nx.sigma(G)


        #omega = 0
        feature_list['omega']=nx.omega(G)

        self.features = feature_list
