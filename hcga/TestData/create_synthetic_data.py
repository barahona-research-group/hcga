#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:17:21 2019

@author: robert
"""

import networkx as nx
import random as rd

def synthetic_data():
    
    graphs=[nx.planted_partition_graph(1, 10, rd.uniform(0.6,1), rd.uniform(0.1,0.4), seed=None, directed=False)
           for i in range(50)]+[nx.planted_partition_graph(2, 5, rd.uniform(0.6,1), rd.uniform(0.1,0.4),
                                                           seed=None, directed=False) for i in range(50)]
   
    for i in range(len(graphs)):
       if not nx.is_connected(graphs[i]):
           if i < 50:
               graphs[i] = nx.planted_partition_graph(1, 10, rd.uniform(0.6,1), rd.uniform(0.1,0.4), seed=None, directed=False)
           elif i > 50:
               graphs[i] = nx.planted_partition_graph(2, 5, rd.uniform(0.6,1), rd.uniform(0.1,0.4), seed=None, directed=False)
        
    
    graph_class=[1 for i in range(50)]+[2 for i in range(50)]
    
    return graphs,graph_class