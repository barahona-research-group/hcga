#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 12:12:56 2020

@author: robert
"""
#%%
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
import pandas as pd
import numpy as np
import scipy as sc
#%%
pandas2ri.activate()


ro.r['load']("/home/robert/Documents/PythonCode/hcga/examples/datasets/Sociomatrix_Data.Rdata")

#%% load meta data

meta_data = pd.read_csv('/home/robert/Documents/PythonCode/hcga/examples/datasets/Network_Metadata.csv',encoding = "ISO-8859-1")

#%%
label_types = meta_data['Network_Type'].unique()
label_id = np.arange(len(label_types))

#%%


labels = []
graphs = []
label_names = []
for i in range(len(ro.r['sociomatrix_data_list'])):

    # ignoring feature node information
    g = ro.conversion.rpy2py(ro.r['sociomatrix_data_list'][i][0])
    
    #
    graphs.append(g)
    label_type = meta_data['Network_Type'][i]
    
    labels.append(np.where(label_types==label_type)[0][0])
    label_names.append(label_type)
    
#%%

from hcga.graph import Graph, GraphCollection

# create graph collection object
g_c = GraphCollection()


# looping over each graph and appending it to the graph collection object
for i,A in enumerate(graphs):
    
    # generating a sparse matrix
    sA = sc.sparse.coo_matrix(A)
    
    # extracting edge list from scipy sparse matrix
    edges = np.array([sA.row,sA.col,sA.data]).T
    
    # passing edge list to pandas dataframe
    edges_df = pd.DataFrame(edges, columns = ['start_node', 'end_node', 'weight'])

    # creating node ids based on size of adjancency matrix
    nodes = np.arange(0,A.shape[0])
    
    # loading node ids into dataframe
    nodes_df = pd.DataFrame(index=nodes)

    # each node should have the same number of node features across all graphs
    # converting node features array to list such that each node is assigned a list.
    #nodes_df['attributes'] = node_features[i].tolist()

    # extracting graph label from labels
    graph_label = labels[i]
    graph_label_name = label_names[i]
    
    # create a single graph object
    graph = Graph(nodes_df, edges_df, graph_label, graph_label_name)

    # add new graph to the collection
    g_c.add_graph(graph)

#%%

print('There are {} graphs'.format(len(g_c.graphs)))
print('There are {} features per node'.format(g_c.get_n_node_features()))


#%%
from hcga.io import save_dataset

save_dataset(g_c, 'custom_dataset_misc', folder='./datasets')

#%%
from hcga.hcga import Hcga

# define an object
h = Hcga()

h.graphs = g_c

h.extract(mode='slow',n_workers=4,timeout=20)

# saving all features into a pickle
h.save_features('./results/custom_dataset_misc/all_features.pkl')

#%%

from hcga.hcga import Hcga

# define an object
h = Hcga()


h.analyse_features(feature_file='/home/robert/Documents/PythonCode/hcga/examples/results/custom_dataset_misc/all_features.pkl',graph_removal=0.5,)

#%%
from hcga.io import load_features
from hcga.analysis import analysis
import pickle
from hcga.analysis import classify_pairwise

features, features_info, graphs = load_features(
        filename="/home/robert/Documents/PythonCode/hcga/examples/results/custom_dataset_misc/all_features.pkl")

accuracy_matrix, top_features, n_pairs = classify_pairwise(
    features, "XG", reduce_set=True, graph_removal=0.5,
)
pickle.dump([top_features, n_pairs], open("top_features_reduced.pkl", "wb"))
pickle.dump(accuracy_matrix, open("accuracy_matrix_reduced.pkl", "wb"))

