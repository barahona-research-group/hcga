#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from hcga.graphs import Graphs
import numpy as np
import os,sys

print("Load graphs...")
dataset = sys.argv[-1]

g = Graphs(directory='./datasets/', dataset=dataset)

#graph_subset = np.arange(0,len(g.graphs), 10)
#g.graphs = [g.graphs[i] for i in graph_subset]
#g.graph_labels = [g.graph_labels[i] for i in graph_subset]

g.load_feature_matrix(filename = 'Output_' + dataset + '/feature_matrix.pkl' )

print("Run classification...")
g.normalise_feature_data()
g.graph_classification(image_folder ='Output_' + dataset, reduc_threshold = 0.15) #threshold is used to compute the reduced set (1 = full set, 0 = no features)
