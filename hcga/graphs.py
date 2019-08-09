import numpy as np
import pandas as pd
import networkx as nx

from hcga.utils import read_graphfile
from hcga.Operations.operations import Operations

from tqdm import tqdm
import time


from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

class Graphs():

    """
        Takes a list of graphs
    """

    def __init__(self, graphs = [], graph_meta_data = [], node_meta_data = [], graph_class = [], directory='', dataset = 'synthetic'):
        self.graphs = graphs # A list of networkx graphs
        self.graph_labels = graph_class # A list of class IDs - A single class ID for each graph.

        self.graph_metadata = graph_meta_data # A list of vectors with additional feature data describing the graph
        self.node_metadata = node_meta_data # A list of arrays with additional feature data describing nodes on the graph

        self.dataset = dataset
        if not graphs:
            self.load_graphs(directory=directory,dataset=dataset)




    def load_graphs(self, directory = '', dataset = 'synthetic'):

        # node labels are within the graph file
        
        # Node features are stored in: G.node[0]['feat'] 
        # Node labels are stored in: G.node[0]['label']
        
        if dataset == 'ENZYMES':
        
            graphs,graph_labels = read_graphfile(directory,dataset)
    
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
    
        elif dataset == 'synthetic':
            from hcga.TestData.create_synthetic_data import synthetic_data
            graphs,graph_labels = synthetic_data()
            
        self.graphs = graphs
        self.graph_labels = graph_labels


    def calculate_features(self):
        """
        Calculation the features for each graph in the set of graphs

        """

        graph_feature_set = []

        cnt = 0
        for G in tqdm(self.graphs):
            print("-------------------------------------------------")              

            print("----- Computing features for graph "+str(cnt)+" -----")               
            start_time = time.time()    
            print("-------------------------------------------------")               

            G_operations = Operations(G)
            G_operations.feature_extraction()
            graph_feature_set.append(G_operations)
            print("-------------------------------------------------")               
            print("Time to calculate all features for graph "+ str(cnt) +": --- %s seconds ---" % round(time.time() - start_time,3))               
            cnt = cnt+1
            
            
        self.graph_feature_set = graph_feature_set
        
        
        
        feature_names=graph_feature_set[0]._extract_data()[0]
        # Create graph feature matrix
        feature_vals_matrix=np.empty([len(graph_feature_set),3*len(feature_names)])  #array([graph_feature_set[0]._extract_data()[1]])
        # Append features for each graph as rows   
        
        
        
        for i in range(0,len(graph_feature_set)):            

            N = self.graphs[i].number_of_nodes()
            E = self.graphs[i].number_of_edges()
            graph_feats = np.asarray(graph_feature_set[i]._extract_data()[1])
            compounded_feats = np.hstack([graph_feats,graph_feats/N,graph_feats/E]) 
            
            
            
            feature_vals_matrix[i,:]=compounded_feats
        
        compounded_feature_names = feature_names + [s +'_N' for s in feature_names] + [s +'_E' for s in feature_names]
        
        raw_feature_matrix=pd.DataFrame(feature_vals_matrix,columns=compounded_feature_names)
        
        self.raw_feature_matrix = raw_feature_matrix 
        
        # remove infinite and nan columns
        feature_matrix_clean = raw_feature_matrix.replace([np.inf, -np.inf], np.nan).dropna(1,how="any")
        
        #remove columns with all zeros
        feats_all_zeros = (feature_matrix_clean==0).all(0)        
        feature_matrix_clean = feature_matrix_clean.drop(columns=feats_all_zeros[feats_all_zeros].index)
        
        # introduce a measure of how many features were removed and their ids and names.
        
        
        
        self.graph_feature_matrix = feature_matrix_clean


        
    def extract_feature(self,n):
        
        graph_feature_matrix = self.graph_feature_matrix
        
        # if n is an int, extract column corresponding to n
        if type(n)==int:
            return graph_feature_matrix.iloc[:,n]
        # if n is a feature name, extract column corresponding to that feature name
        else:
            return graph_feature_matrix[n]   
        

    def top_features_model(self,method='random_forest'):
        
        """
        This class identifies top features given a classification algorithm 
        that ranks features.
        
        """
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn.feature_selection import SelectFromModel
        from sklearn.metrics import accuracy_score
                    
        self.normalise_feature_data()
        X=self.X_N
        y=self.y
        feature_names=[col for col in self.graph_feature_matrix.columns]
        
        if method =='random_forest': 
            
            top_features_list=[]
            
            for i in range(5):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            
                clf=RandomForestClassifier(n_estimators=10000, random_state=0)
                clf.fit(X_train, y_train)
            
                sfm = SelectFromModel(clf, threshold=0.01)
                sfm.fit(X_train, y_train)
            
                for feature_list_index in sfm.get_support(indices=True):
                    top_features_list.append(feature_names[feature_list_index])
            
            top_features_list=list(dict.fromkeys(top_features_list))
            
            self.top_features_list=top_features_list
        
        elif method == 'univariate':
            return  
        
            
        return

        
    def top_features_univariate(self,method='random_forest', plot=True):
        
        from sklearn.svm import LinearSVC
        from sklearn import datasets
        from sklearn import model_selection
        from sklearn.feature_selection import SelectFromModel
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import balanced_accuracy_score               
        import seaborn as sns

        seed = 7
        
        scoring = 'balanced_accuracy'
        
        X=self.X_N
        y=self.y
        feature_names=[col for col in self.graph_feature_matrix.columns]
        
        accuracy = []        
        for i in range(0,X.shape[1]):
            print('Computing feature: '+str(i))
            kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(LinearSVC(), X[:,i].reshape(-1,1), y, cv=kfold, scoring=scoring)
            accuracy.append(cv_results.mean())
        
        top_feat_ids = np.flip(np.argsort(accuracy))
        top_feat_acc = np.flip(np.sort(accuracy))
        top_feats_names = [feature_names[i] for i in top_feat_ids]
        
        self.top_feat_ids =  top_feat_ids
        self.top_feat_acc = top_feat_acc
        self.top_feats_names = top_feats_names
        
        if plot is True:
            plt.figure()
            plt.hist(top_feat_acc,20)
        
        
        # plot top 40 feats
        top_feats_names[:40]
        df_top40 = pd.DataFrame(columns = top_feats_names[:40], data=X[:,top_feat_ids[:40]])    
        
        
        cor = np.abs(df_top40.corr())
        
        from scipy.cluster.hierarchy import dendrogram, linkage
        Z = linkage(cor, 'ward')
        
        plt.figure()
        dn = dendrogram(Z)         

        
        new_index = [int(i) for  i in dn['ivl']]
        top_feats_names = [top_feats_names[i] for i in new_index]        
        
        df = df_top40[top_feats_names]
        cor2 = np.abs(df.corr())
        
        plt.figure()
        ax = sns.heatmap(cor2, linewidth=0.5)
        plt.show()
            
        
        
    
    def organise_top_features(self):
            
        top_features_list = self.top_features_list
            
        feature_values = np.transpose(np.array([self.extract_feature(i).values for i in top_features_list]))
            
        top_feature_matrix = pd.DataFrame(feature_values,columns=top_features_list)
            
        t_X = top_feature_matrix.as_matrix()
            
        t_X_N = t_X / t_X.max(axis=0)
        t_X_N = t_X_N[:,~np.isnan(t_X_N).any(axis=0)]
            
        self.top_feature_matrix = top_feature_matrix
        self.t_X_N = t_X_N
        self.y = np.asarray(self.graph_labels)
            




    def normalise_feature_data(self):

        graph_feature_matrix = self.graph_feature_matrix
        
        X=graph_feature_matrix.as_matrix()

        X_N = X / X.max(axis=0)
        
        self.X_N = X_N
        self.y=np.asarray(self.graph_labels)

        return



    def graph_classification(self,plot=True):

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

        if plot:
            fig = plt.figure()
            fig.suptitle('Algorithm Comparison')
            ax = fig.add_subplot(111)
            plt.boxplot(results)
            plt.ylim(0,1)
            ax.set_xticklabels(names)
        #plt.show()

        return results
    

    def save_feature_set(self,filename = 'TestData/feature_set.pkl'):
        import pickle as pkl        
        feature_matrix = self.graph_feature_matrix
        
        with open(filename,'wb') as output:
            pkl.dump(feature_matrix,output,pkl.HIGHEST_PROTOCOL)
            
        
        
    
    def load_feature_set(self,filename = 'TestData/feature_set.pkl'):
        import pickle as pkl
        with open(filename,'rb') as output:
            feature_matrix = pkl.load(output)
        
        self.graph_feature_matrix = feature_matrix

            