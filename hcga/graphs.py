import numpy as np
import pandas as pd
import networkx as nx

from hcga.utils import read_graphfile
from hcga.Operations.operations import Operations

from tqdm import tqdm

from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

class Graphs():

    """
        Takes a list of graphs
    """

    def __init__(self, graphs = [], graph_meta_data = [], node_meta_data = [], graph_class = []):
        self.graphs = graphs # A list of networkx graphs
        self.graph_labels = graph_class # A list of class IDs - A single class ID for each graph.

        self.graph_metadata = graph_meta_data # A list of vectors with additional feature data describing the graph
        self.node_metadata = node_meta_data # A list of arrays with additional feature data describing nodes on the graph

        if not graphs:
            self.load_graphs()




    def load_graphs(self, directory = 'TestData', dataname = 'ENZYMES'):

        # node labels are within the graph file
        
        # Node features are stored in: G.node[0]['feat'] 
        # Node labels are stored in: G.node[0]['label']
        
        graphs,graph_labels = read_graphfile(directory,dataname)

        # selected data for testing code
        selected_data = np.arange(0,300,1)
        graphs = [graphs[i] for i in list(selected_data)]
        graph_labels = [graph_labels[i] for i in list(selected_data)]
        
        
        to_remove = []
        for i,G in enumerate(graphs): 
            if not nx.is_connected(G):  
                print('Graph '+str(i)+' is not connected. Taking largest subgraph and relabelling the nodes.')
                Gc = max(nx.connected_component_subgraphs(G), key=len)
                mapping=dict(zip(Gc.nodes,range(0,len(Gc))))
                Gc = nx.relabel_nodes(Gc,mapping)                
                graphs[i] = Gc
            
            if len(graphs[i])<3:
                to_remove.append(i)
         
        # removing graphs with less than 2 nodes
        graph_labels = [i for j, i in enumerate(graph_labels) if j not in to_remove]
        graphs = [i for j, i in enumerate(graphs) if j not in to_remove]

        
        self.graphs = graphs
        self.graph_labels = graph_labels


    def calculate_features(self):
        """
        Calculation the features for each graph in the set of graphs

        """

        graph_feature_set = []

        for G in tqdm(self.graphs):
            G_operations = Operations(G)
            G_operations.feature_extraction()
            graph_feature_set.append(G_operations)
            
        feature_names=graph_feature_set[0]._extract_data()[0]
        # Create graph feature matrix
        feature_vals_matrix=np.array([graph_feature_set[0]._extract_data()[1]])
        # Append features for each graph as rows
        for i in range(1,len(graph_feature_set)):
            feature_vals_matrix=np.vstack([feature_vals_matrix,graph_feature_set[i]._extract_data()[1]])
        
        graph_feature_matrix=pd.DataFrame(feature_vals_matrix,columns=feature_names)
        
        # Find the position of nan or inf within features
        nan_inf_pos=[]
        for j in range(len(graph_feature_set)):
            nan_inf_pos.append(np.where(np.isnan(graph_feature_set[j]._extract_data()[1])==True)[0])
            nan_inf_pos.append(np.where(np.isinf(graph_feature_set[j]._extract_data()[1])==True)[0])
        nan_inf_positions=[r for s in nan_inf_pos for r in s]
        # Remove any duplicates
        nan_inf_positions=list(dict.fromkeys(nan_inf_positions))
        
        # Features to delete
        deleted_features=[feature_names[k] for k in nan_inf_positions]
        # Delete columns where nan or inf are present
        for k in range(len(deleted_features)):
            del graph_feature_matrix[deleted_features[k]]

        
        self.deleted_features = deleted_features
        self.graph_feature_matrix = graph_feature_matrix
        self.calculated_graph_features = graph_feature_set
        
        
    def extract_feature(self,n):
        
        graph_feature_matrix = self.graph_feature_matrix
        
        # if n is an int, extract column corresponding to n
        if type(n)==int:
            return graph_feature_matrix.iloc[:,n]
        # if n is a feature name, extract column corresponding to that feature name
        else:
            return graph_feature_matrix[n]   
        

    def top_features(self,method='random_forest'):

        if method =='random_forest':
            return
        elif method == 'univariate':
            return  
        
            
            
        return



    """def organise_feature_data(self):

        X = []
        for graph in self.calculated_graph_features:
            feature_names, features = graph._extract_data()
            X.append(features)

        X = np.asarray(X)
        y = np.asarray(self.graph_labels)

        self.X = X
        self.y = y

        return"""



    def normalise_feature_data(self):

        graph_feature_matrix=self.graph_feature_matrix
        
        X=graph_feature_matrix.as_matrix()

        X_N = X / X.max(axis=0)
        
        """
        We remove nan but we haven't removed the feature names that were nan
        """
        X_N = X_N[:,~np.isnan(X_N).any(axis=0)] # removing features with nan
        
    
        
        self.X_N = X_N
        self.y=np.asarray(self.graph_labels)

        return

    def save_obj(self):
        import pickle as pkl
        with open('dataset.pkl','wb') as output:
            pkl.dump(self,output,pkl.HIGHEST_PROTOCOL)
            
        
    
    def load_obj(self):
        import pickle as pkl
        with open('dataset.pkl','wb') as output:
            pkl.load(output)

    def graph_classification(self):

        """
        Graph Classification

        """

        from sklearn import model_selection
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier

        """self.organise_feature_data()"""
        self.normalise_feature_data()

        X = self.X_N
        y = self.y


        # prepare configuration for cross validation test harness
        seed = 7

        # prepare models
        models = []

        models.append(('RandomForest', RandomForestClassifier()))
        models.append(('LinearSVM', LinearSVC()))

        results = []
        names = []
        scoring = 'accuracy'
        for name, model in models:
            	kfold = model_selection.KFold(n_splits=10, random_state=seed)
            	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
            	results.append(cv_results)
            	names.append(name)
            	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            	print(msg)

        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        #plt.show()

        return
