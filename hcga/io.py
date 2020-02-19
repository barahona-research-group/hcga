"""input/output functions"""
import sys
import os
import pickle
import csv

def _ensure_weights(graphs):
    """ensure that graphs edges have a weights value"""
    for i, graph in enumerate(graphs): 
        for u, v in graph.edges: 
            if 'weight' not in graph[u][v]:
                graph[u][v]['weight'] = 1.


def _remove_small_graphs(graphs, n_node_min=2):
    """remove too small graphs"""
    return [graph for graph in graphs if len(graph) > n_node_min]


def load_graphs(graph_dataset, location='datasets'):
   """load graphs in a list of networkx graphs"""
   graphs = getattr(sys.modules[__name__], "load_%s" % graph_dataset)(location)

   graphs = _remove_small_graphs(graphs)
   _ensure_weights(graphs)
    
   return graphs


def load_neurons(location):
    """load the neuron dataset (from Kanari et al. paper)"""
    graphs_full = pickle.load(open(os.path.join(location, 'neurons', 'neurons_animals_for_hcga.pkl'), 'rb'))

    graphs = []
    graph_labels = []
    for i in range(len(graphs_full)):
        graph = graphs_full[i][0]
        graph.name = graphs_full[i][1]
        graphs.append(graph)
    
    return graphs


def save_features(feature_matrix, feature_info, filename='features.pkl'):
    """Save the features in a pickle"""
    pickle.dump([feature_matrix, feature_info], open(filename,'wb'))







###old functions###









def old():
    if dataset == 'ENZYMES' or dataset == 'DD' or dataset == 'COLLAB' or dataset == 'PROTEINS' or dataset == 'REDDIT-MULTI-12K' or dataset == 'ENZYMES1':
        
        if dataset == 'ENZYMES1':
            graphs,graph_labels = read_graphfile(directory,'ENZYMES')
        else:
            graphs,graph_labels = read_graphfile(directory,dataset)

        to_remove = []
        for i,G in enumerate(graphs): 

            #hack to add weights on edges if not present
            for u,v in G.edges: 
                if len(G[u][v]) == 0:
                    G[u][v]['weight'] = 1.
                
            if len(graphs[i])<3:
                to_remove.append(i)
         
        # removing graphs with less than 2 nodes
        graph_labels = [i for j, i in enumerate(graph_labels) if j not in to_remove]
        graphs = [i for j, i in enumerate(graphs) if j not in to_remove]

        
    
    elif dataset == 'synthetic':
        from hcga.TestData.create_synthetic_data import synthetic_data
        graphs,graph_labels = synthetic_data()

    elif dataset == 'synthetic_watts_strogatz':
        from hcga.TestData.create_synthetic_data import synthetic_data_watts_strogatz
        graphs,graph_labels = synthetic_data_watts_strogatz(N=1000)

    elif dataset == 'synthetic_powerlaw_cluster':
        from hcga.TestData.create_synthetic_data import synthetic_data_powerlaw_cluster
        graphs,graph_labels = synthetic_data_powerlaw_cluster(N=1000)

    elif dataset == 'synthetic_sbm':
        from hcga.TestData.create_synthetic_data import synthetic_data_sbm
        graphs,graph_labels = synthetic_data_sbm(N=1000)

    elif dataset == 'HELICENES':
        graphs,graph_labels = pickle.load(open(directory+'/HELICENES/helicenes_for_hcga.pkl','rb'))

    elif dataset == 'NEURONS':
        graphs_full = pickle.load(open(directory+'/NEURONS/neurons_animals_for_hcga.pkl','rb'))

        graphs = []
        graph_labels = []
        for i in range(len(graphs_full)):
            G = graphs_full[i][0]
            graphs.append(graphs_full[i][0])
            graph_labels.append(graphs_full[i][1])
        
    self.graphs = graphs
    self.graph_labels = graph_labels















def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
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


