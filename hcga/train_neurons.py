#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:31:41 2019

@author: robert
"""

# test script that loads some enzymes data and runs it

from hcga.graphs import Graphs
import numpy as np
import os
import pickle
import sys 

cwd = os.getcwd()
#g = Graphs(directory=cwd+'/TestData',dataset='COLLAB')



filehandler = open('TestData/NEURONS/neurons_for_hcga.pkl','rb')
graphs_full = pickle.load(filehandler)

graphs = []
graph_labels = []


for i in range(len(graphs_full)):
    G = graphs_full[i][0]
    if G.number_of_nodes() > 0:	
        graphs.append(graphs_full[i][0])
        graph_labels.append(graphs_full[i][1])

graph_labels = np.asarray(graph_labels)

len(graphs)

g = Graphs(graphs=graphs)
g.graph_labels = graph_labels


for i in range(10):
    print(len(g.graphs[i]))

g.n_processes = 20

g.calculate_features(calc_speed='fast',parallel=False)

g.normalise_feature_data()

g.graph_classification()
