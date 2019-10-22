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


cwd = os.getcwd()
print("Load graphs...")
g = Graphs(directory=cwd+'/TestData',dataset='ENZYMES')
g.n_processes = 80
graph_subset = np.arange(0,len(g.graphs), 1)

g.graphs = [g.graphs[i] for i in graph_subset]
g.graph_labels = [g.graph_labels[i] for i in graph_subset]

print("Calculate features...")
g.calculate_features(calc_speed='slow', parallel = True)

g.normalise_feature_data()

if not os.path.isdir('Images'):
    os.mkdir('Images')

g.graph_classification()
