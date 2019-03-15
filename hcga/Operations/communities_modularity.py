# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np

from networkx.algorithms.community import greedy_modularity_communities


class ModularityCommunities():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):

        self.feature_names = ['num_comms_greedy_mod']

        G = self.G

        feature_list = []

        # basic normalisation parameters
        N = G.number_of_nodes()
        E = G.number_of_edges()

        # The optimised number of communities using greedy modularity
        feature_list.append(len(greedy_modularity_communities(G)))

        self.features = feature_list
