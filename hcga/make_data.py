#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:17:21 2019

@author: robert
"""

import networkx as nx
import random as rd
import numpy as np
import re
import wget

import os
import zipfile
import pickle 


def unzip(zip_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall()    
 
    
def save_dataset(graphs, labels, filename):
    """Save the features in a pickle"""
    with open(filename, "wb") as f:
        pickle.dump([graphs,labels], f)
    


def load_dataset(filename):
    """Save the features in a pickle"""
    with open(filename, "rb") as f:
        [graphs, labels] = pickle.load(f)
    return graphs, labels
    


def make_benchmark_dataset(data_name='ENZYMES'):
    """
    Standard datasets include:
        DD
        ENZYMES
        REDDIT-MULTI-12K
        PROTEINS
        MUTAG
    """
    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{}.zip'.format(data_name)
    wget.download(url)
    unzip('{}.zip'.format(data_name))
    os.remove('{}.zip'.format(data_name))    
    graphs,labels = read_graphfile(data_name,data_name)    
    save_dataset(graphs, labels, '{}.pkl'.format(data_name))

    
def make_test_data(add_features=False, save_data=False):
    """ Makes pickle with graphs that test robustness of hcga """
    
    graphs = []
    
    #one, two and three node graphs
    graphs.append(_add_graph_desc(nx.grid_graph([1]), 
        'one-node graph'))    
    graphs.append(_add_graph_desc(nx.grid_graph([2]), 
        'two-node graph'))    
    graphs.append(_add_graph_desc(nx.grid_graph([3]), 
        'three-node graph'))    
    
    # no edges
    G = nx.Graph()
    G.add_node(0); G.add_node(1, weight=2); G.add_node(2, weight=3)
    graphs.append(_add_graph_desc(G, 'graph without edges'))
    
    # directed graph no weights
    G = nx.DiGraph()
    G.add_nodes_from(range(100, 110))
    graphs.append(_add_graph_desc(G, 'directed graph with no weights'))
    
    # directed graph weighted
    G = nx.DiGraph(); H = nx.path_graph(10); G.add_nodes_from(H)
    G.add_edges_from(H.edges)
    graphs.append(_add_graph_desc(G, 'directed graph weighted'))
    
    # multigraph

    # adding features to all
    if add_features:
        graphs = [add_dummy_node_features(graph) for graph in graphs]
    
    labels = np.arange(len(graphs))
    
    if save_data:
        save_dataset(graphs, labels, 'test_data.pkl')
    
    return graphs, labels

def _add_graph_desc(g, desc):
    """Add descrition desc to graph g as a graph attribute"""
    g.graph['description'] = desc
    return g
    
def add_dummy_node_features(graph):    
    for u in graph.nodes():
        graph.nodes[u]['feat'] = np.random.rand(10)    
    return graph




def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                node_labels+=[int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')
 
    filename_node_attrs=prefix + '_node_attributes.txt'
    node_attrs=[]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')
       
    label_has_zero = False
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]

    # assume that all graph labels appear in the dataset 
    #(set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            #if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    #graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    #if label_has_zero:
    #    graph_labels += 1
    
    filename_adj=prefix + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            index_graph[graph_indic[e0]]+=[e0,e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]

    graphs=[]
    node_label_list = []
    for i in range(1,1+len(adj_list)):
        # indexed from 1 here
        G=nx.from_edgelist(adj_list[i])
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue
      
        # add features and labels
        G.graph['label'] = graph_labels[i-1]
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u-1]
                node_label_one_hot[node_label] = 1
                G.nodes[u]['label'] = node_label_one_hot		
            if len(node_attrs) > 0:
                G.nodes[u]['feat'] = node_attrs[u-1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping={}
        it=0
        if float(nx.__version__)<2.0:
            for n in G.nodes():
                mapping[n]=it
                it+=1
        else:
            for n in G.nodes:
                mapping[n]=it
                it+=1
            
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    return graphs, graph_labels



























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


def synthetic_data_watts_strogatz(N=1000):
    
    graphs = []
    graph_labels = []
    
    p = np.linspace(0,1,N)
    
    for i in range(N):
        G = nx.connected_watts_strogatz_graph(40,5,p[i])
        graphs.append(G)
        graph_labels.append(p[i])
    
    return graphs, np.asarray(graph_labels)

def synthetic_data_powerlaw_cluster(N=1000):    
    graphs = []
    graph_labels = []
    
    p = np.linspace(0,1,N)
    
    for i in range(N):
        G = nx.powerlaw_cluster_graph(40,5,p[i])
        graphs.append(G)
        graph_labels.append(p[i])
    
    return graphs, np.asarray(graph_labels)

def synthetic_data_sbm(N=1000):
    graphs = []
    graph_labels = [] 
    
    import random
    
    
    for i in range(int(N/2)):
        G = nx.stochastic_block_model([random.randint(10,30),random.randint(10,30),random.randint(10,30)],[[0.6,0.1,0.1],[0.1,0.6,0.1],[0.1,0.1,0.6]])
        graphs.append(G)
        graph_labels.append(1)
    
    for i in range(int(N/2)):
        G = nx.stochastic_block_model([random.randint(20,40),random.randint(20,40)],[[0.6,0.1],[0.1,0.6]])
        graphs.append(G)
        graph_labels.append(2)
    
    return graphs, np.asarray(graph_labels)


    
