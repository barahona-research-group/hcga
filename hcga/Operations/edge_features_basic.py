#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:43:56 2019

@author: Henry
"""

import networkx as nx

from hcga.Operations.utils import summary_statistics

class EdgeFeaturesBasic():
    
    
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}
        
    def feature_extraction(self):
        
        G = self.G
        
        
        feature_list = {}
        

        edge_attributes = list(nx.get_edge_attributes(G,'weight').values())

        feature_list = summary_statistics(feature_list,edge_attributes,'edge_weights')       
        
        
        self.features=feature_list