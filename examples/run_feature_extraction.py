#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from hcga.graphs import Graphs
import numpy as np
import os, sys

print("Load graphs...")
dataset = sys.argv[-1]

g = Graphs(directory='./datasets', dataset=dataset)

g.n_processes = 80

#graph_subset = np.arange(0,len(g.graphs), 10)
#g.graphs = [g.graphs[i] for i in graph_subset]
#g.graph_labels = [g.graph_labels[i] for i in graph_subset]

print("Calculate features...")
g.calculate_features(calc_speed='fast', parallel = True)

if not os.path.isdir('Output_' + dataset):
    os.mkdir('Output_' + dataset)

#save the features
g.save_feature_matrix(filename = 'Output_' + dataset + '/feature_matrix.pkl' )
