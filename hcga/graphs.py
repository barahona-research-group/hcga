# -*- coding: utf-8 -*-
# This file is part of hcga.
#
# Copyright (C) 2019, 
# Robert Peach (r.peach13@imperial.ac.uk), 
# Alexis Arnaudon (alexis.arnaudon@epfl.ch), 
# https://github.com/ImperialCollegeLondon/hcga.git
#
# hcga is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# hcga is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with hcga.  If not, see <http://www.gnu.org/licenses/>.


#to remove warnings
import numpy
numpy.seterr(all='ignore') 
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import pickle as pickle
import networkx as nx

from hcga.utils import read_graphfile
from hcga.Operations.operations import Operations

from tqdm import tqdm
import time

from multiprocessing import Pool
from functools import partial

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

#fix RNGs for reproducibility
import random
random.seed(10)
np.random.seed(10)

class Graphs():

    """
        Main class of hcga, with all the construction analysis functions. 
        The constructor takes either a list of networkx graphs, or a directory with a list of graph, and the name of the dataset (if it exists). 

        Parameters
        ----------

        graphs: list 
            list of graphs
        directory: folder
            directory with graphs
        dataset: string
            name of the dataset to load the graphs

    """

    def __init__(self, graphs = None, graph_meta_data = [], node_meta_data = [], graph_class = [], directory='', dataset = 'synthetic'):
        
        self.graphs = graphs # A list of networkx graphs
        self.graph_labels = graph_class # A list of class IDs - A single class ID for each graph.

        self.graph_metadata = graph_meta_data # A list of vectors with additional feature data describing the graph
        self.node_metadata = node_meta_data # A list of arrays with additional feature data describing nodes on the graph
        self.n_processes = 4
        self.dataset = dataset
        
        if not graphs:
            self.load_graphs(directory=directory,dataset=dataset)




    def load_graphs(self, directory = '', dataset = 'synthetic'):
        """
        Function to load graphs from a directory. 

        Parameters
        ----------
        directory: string
            directory path with graphs
        dataset: string
            Name of the dataset, the following are allowed: 
                * ENZYMES: benchmark 
                * DD: benchmark 
                * COLLAB: benchmark
                * PROTEINS: benchmark 
                * REDDIT-MULTI-12K: benchmark
                * synthetic: contains several variants of synthetics 
                * HELICENES: Julia Schmidt's dataset
                * NEURONS: neurons dataset for different animals 
        
        """

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
                    
                """
                The fact that some nodes are not connected needs to be changed. We need to append the subgraphs and run feature extraction
                on each subgraph. Add features that relate to the extra subgraphs. or features indicating there are subgraphs.
                """
#                if not nx.is_connected(G):  
#                    print('Graph '+str(i)+' is not connected. Taking largest subgraph and relabelling the nodes.')
#                    Gc = max(nx.connected_component_subgraphs(G), key=len)
#                    mapping=dict(zip(Gc.nodes,range(0,len(Gc))))
#                    Gc = nx.relabel_nodes(Gc,mapping)                
#                    graphs[i] = Gc
                
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

    
    def calculate_features(self,calc_speed='slow',parallel=True):
        """
        Extract the features from each graph in the set of graphs

        Parameters
        ----------

        calc_speed: string
            set of features to consider (from the operations.csv file). Can take 'slow', 'medium' or 'fast'. 
        parallel: bool
            True to run with multiprocessing 

        """

        if parallel:
            calculate_features_single_graphf = partial(calculate_features_single_graph, calc_speed)
            
            with Pool(processes = self.n_processes) as p_feat:  #initialise the parallel computation
                self.graph_feature_set = list(tqdm(p_feat.imap(calculate_features_single_graphf, self.graphs), total = len(self.graphs)))
            
            comp_times = []
            for op in self.graph_feature_set:
                comp_times.append(list(op.computational_times.values()))
            comp_times_mean = np.mean(np.array(comp_times), axis = 0)

            #print the computational time from fast to slow
            sort_id = np.argsort(comp_times_mean)
            fns = [] 
            for i, fn in enumerate(list(self.graph_feature_set[0].computational_times.keys())):
                fns.append(fn)

            for i in sort_id:
                print('Computation time for feature: ' + str(fns[i]) + ' is ' + str(np.round(comp_times_mean[i],3)) + ' seconds.')
                
                
        else: 
            graph_feature_set = []
            cnt = 0
            for G in tqdm(self.graphs):
                print("-------------------------------------------------")              
    
                print("----- Computing features for graph "+str(cnt)+" -----")               
                start_time = time.time()    
                print("-------------------------------------------------")               
    
                G_operations = Operations(G)
                G_operations.feature_extraction(calc_speed=calc_speed)
                graph_feature_set.append(G_operations)
                print("-------------------------------------------------")               
                print("Time to calculate all features for graph "+ str(cnt) +": --- %s seconds ---" % round(time.time() - start_time,3))               
                cnt = cnt+1
                self.graph_feature_set_temp = graph_feature_set
            self.graph_feature_set = graph_feature_set            

        feature_names = self.graph_feature_set[0]._extract_data()[0]
            
        # Create graph feature matrix
        feature_vals_matrix = np.empty([len(self.graph_feature_set),3*len(feature_names)])  
        
        # Append features for each graph as rows   
        for i in range(len(self.graph_feature_set)):            

            N = self.graphs[i].number_of_nodes()
            E = self.graphs[i].number_of_edges()
            graph_feats = np.asarray(self.graph_feature_set[i]._extract_data()[1])
            compounded_feats = np.hstack([graph_feats, graph_feats/N,graph_feats/E]) 
           
            debug_features = False
            if debug_features:
                namesf =  self.graph_feature_set[i]._extract_data()[0]
                for ci, cf in enumerate( self.graph_feature_set[i]._extract_data()[1]):
                    print(namesf[ci], cf)

            feature_vals_matrix[i,:] = compounded_feats
        
        compounded_feature_names = feature_names + [s +'_N' for s in feature_names] + [s +'_E' for s in feature_names]
        
        raw_feature_matrix = pd.DataFrame(feature_vals_matrix, columns = compounded_feature_names)
        
        self.raw_feature_matrix = raw_feature_matrix 
        print('Number of raw features: ', np.shape( raw_feature_matrix )[1])

        # remove infinite and nan columns
        feature_matrix_clean = raw_feature_matrix.replace([np.inf, -np.inf], np.nan).dropna(1,how="any")
        print('Number of features without nans/infs: ', np.shape( feature_matrix_clean)[1])
        
        #remove columns with all zeros
        feats_all_zeros = (feature_matrix_clean==0).all(0)        
        feature_matrix_clean = feature_matrix_clean.drop(columns=feats_all_zeros[feats_all_zeros].index)
        
        # remove features with constant values
        feature_matrix_clean = feature_matrix_clean.loc[:, (feature_matrix_clean != feature_matrix_clean.iloc[0]).any()]
        
        self.graph_feature_matrix = feature_matrix_clean
        print("Final number of features extracted:", np.shape(feature_matrix_clean)[1])


        
    def extract_feature(self,n):
        """
        Extract a feature from the feature matrix

        Parameters
        ----------
        n: int or string
            Either the feature number of its name

        Returns
        -------
        feature: list
            Feature values across graphs

        """ 

        graph_feature_matrix = self.graph_feature_matrix
        
        # if n is an int, extract column corresponding to n
        if type(n)==int:
            return graph_feature_matrix.iloc[:,n]
        # if n is a feature name, extract column corresponding to that feature name
        else:
            return graph_feature_matrix[n]   
        
    def normalise_feature_data(self):
        """
        Normalise the feature matrix using sklearn scaler to remove the mean and scale to unit variance
        """

        from sklearn.preprocessing import StandardScaler
        
        graph_feature_matrix = self.graph_feature_matrix
        
        X=graph_feature_matrix.values
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)     
        
        self.X_norm = X_norm
        self.y=np.asarray(self.graph_labels)

    def graph_classification(self,plot=True,ml_model='xgboost', data='all', reduc_threshold = 0.9, image_folder='images'):

        """
        Graph classification
       
        Parameters
        ----------

        data: string
            the type of features to classify
                * 'all' : all the features calculated
                * 'feats' : features based on node features and node labels only
                * 'topology' : features based only on the graph topology features
        ml_model: string
            ML method to use, can be:
                * xgboost
                * random_forest

        """
        
        self.normalise_feature_data()

        X = self.X_norm
        y = self.y


        # defining the feature set
        feature_names=[col for col in self.graph_feature_matrix.columns]            

        if data != 'all':
            matching_NL = [i for i, j in enumerate(feature_names) if 'NL' in j]
            matching_NF = [i for i, j in enumerate(feature_names) if 'NF' in j]
            matching = matching_NL + matching_NF
            
        if data=='topology':
            X = np.delete(X,matching,axis=1)
            feature_names = [i for j, i in enumerate(feature_names) if j not in matching]

        elif data=='feats':
            X = X[:,matching]
            feature_names = [i for j, i in enumerate(feature_names) if j in matching]



        testing_accuracy, top_feats = classification(X,y,ml_model)
        print("Mean test accuracy full set: --- {0:.3f} ---)".format(np.mean(testing_accuracy)))            

        # get names of top features
        top_features_list = top_features(X,top_feats, feature_names, image_folder = image_folder, threshold = reduc_threshold, plot=plot)  
        self.top_feats = top_feats         
        self.top_features_list=top_features_list       
            
        # re running classification with reduced feature set
        X_reduced, top_feat_indices = reduce_feature_set(X,top_feats, threshold = reduc_threshold)        
        testing_accuracy_reduced_set, top_feats_reduced_set = classification(X_reduced,y,ml_model) 
        print("Final mean test accuracy reduced feature set: --- {0:.3f} ---)".format(np.mean(testing_accuracy_reduced_set)))            
        
        # top and violin plot top features
        if plot:
            self.top_features_importance_plot(X,top_feat_indices,feature_names,y, name='xgboost', image_folder = image_folder)
            self.plot_violin_feature(X,y,top_feat_indices[0],feature_names, name='xgboost', image_folder = image_folder)


        # univariate classification on top features
        univariate_topfeat_acc = univariate_classification(X_reduced,y)
        feature_names_reduced = [feature_names[i] for i in top_feat_indices]
        top_feat_index = np.argsort(univariate_topfeat_acc)[::-1]       

        # top and violin plot top features from univariate
        if plot:
            self.top_features_importance_plot(
                X_reduced,top_feat_index[0:2],
                feature_names_reduced,y, name='univariate', 
                image_folder = image_folder)
            self.plot_violin_feature(
                X_reduced,y,top_feat_index[0],feature_names_reduced, 
                name='univariate', image_folder = image_folder)

        self.test_accuracy = testing_accuracy_reduced_set 

        return np.mean(testing_accuracy_reduced_set)
    
    
    
    def graph_regression(self,plot=True, data='all'):
        """
        Perform graph regression
        """ 

        from sklearn.model_selection import StratifiedKFold   
        import xgboost
        from sklearn.metrics import explained_variance_score

               
        self.normalise_feature_data()

        X = self.X_norm
        y = self.y
        
        feature_names=[col for col in self.graph_feature_matrix.columns]            

        X = np.delete(X,np.where(np.isnan(y))[0],axis=0)
        y = np.delete(y,np.where(np.isnan(y))[0],axis=0)
        
        # Let's try XGboost algorithm to see if we can get better results
        xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08,gamma=0,subsample=0.75,colsample_bytree=1, max_depth=7)
                
        bins = np.logspace(np.min(y), np.max(y), 10) 
        y_binned = np.digitize(y, bins)
        
        skf = StratifiedKFold(n_splits = 10, shuffle = True)        
        
        top_feats = []        
        explained_variance = []
        for train_index, test_index in skf.split(X, y_binned):            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
            xgb.fit(X_train,y_train)
        
            y_pred = xgb.predict(X_test)
            
            explained_variance.append(explained_variance_score(y_test,y_pred))

            print("Fold explained variance: --- {0:.3f} ---)".format(explained_variance_score(y_test,y_pred)))            
            top_feats.append(xgb.feature_importances_)

        print("Final mean explained variance: --- {0:.3f} ---)".format(np.mean(explained_variance)))            
        print("Final .std explained variance: --- {0:.3f} ---)".format(np.std(explained_variance)))           

        top_features_list = top_features(X,top_feats,feature_names)            
        self.top_features_list=top_features_list            

            
        if plot==True:
            self.top_features_importance_plot(X,top_feats,y)
    
    def graph_classification_mlp(self,X = None, y = None , verbose=True):
        """
        Classify graphs with MLP algorithm
        """

        #from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelBinarizer        
        from sklearn.model_selection import StratifiedKFold          
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import accuracy_score
        
        
        self.normalise_feature_data()

        if X is None:
            X = self.X_norm
            y = self.y        
       
        testing_accuracy = []
        
        
        counts = np.bincount(y)
        least_populated_class = np.argmin(counts)
        if least_populated_class<10:
            skf = StratifiedKFold(n_splits=len(y[y==least_populated_class]), random_state=10, shuffle=True)
        else:
            skf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
        
        
        for train_index, test_index in skf.split(X, y):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #X_train, y_train, X_test, y_test = load_dataset(X,y)
            
            ## Changing labels to one-hot encoded vector
            if len(np.unique(y_train)) > 2:
                lb = LabelBinarizer()
                y_train = lb.fit_transform(y_train)
                y_test = lb.transform(y_test)
            else:
                y_train = np.array([y_train, 1-y_train]).T
                y_test = np.array([y_test, 1-y_test]).T
        
            print('Train labels dimension:');print(y_train.shape)
            print('Test labels dimension:');print(y_test.shape)            
            

            s = tf.Session()  # Create new session            
            
            ## Defining various initialization parameters for 600-256-128-# MLP model
            num_classes = y_train.shape[1]
            num_features = X_train.shape[1]
            num_output = y_train.shape[1]
            num_layers_0 = 256
            num_layers_1 = 128#128
            starter_learning_rate = 0.001
            regularizer_rate = 0.1#0.1
            
            # Placeholders for the input data
            input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
            input_y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')
            
            ## for dropout layer
            keep_prob = tf.placeholder(tf.float32)
            
            
            weights_0 = tf.Variable(tf.random_normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
            bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
            
            weights_1 = tf.Variable(tf.random_normal([num_layers_0,num_layers_1], stddev=(1/tf.sqrt(float(num_layers_0)))))
            bias_1 = tf.Variable(tf.random_normal([num_layers_1]))
            
            weights_2 = tf.Variable(tf.random_normal([num_layers_1,num_output], stddev=(1/tf.sqrt(float(num_layers_1)))))
            bias_2 = tf.Variable(tf.random_normal([num_output]))
            
            hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
            hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)
            
            hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0_0,weights_1)+bias_1)
            hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)
            
            predicted_y = tf.sigmoid(tf.matmul(hidden_output_1_1,weights_2) + bias_2)
            
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y,labels=input_y)) + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)))
            
            learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
            
            
            ## Adam optimzer for finding the right weight
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,weights_2,bias_0,bias_1,bias_2])
            
            ## Metrics definition
            correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predicted_y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
            ## Training parameters
            batch_size = 128
            epochs=200
            dropout_prob = 0.6
            training_accuracy = []
            training_loss = []
            validation_accuracy = []
            s.run(tf.global_variables_initializer())
            for epoch in range(epochs):    
                arr = np.arange(X_train.shape[0])
                np.random.shuffle(arr)
                for index in range(0,X_train.shape[0],batch_size):
                    s.run(optimizer, {input_X: X_train[arr[index:index+batch_size]],
                                      input_y: y_train[arr[index:index+batch_size]],
                                    keep_prob:dropout_prob})
                training_accuracy.append(s.run(accuracy, feed_dict= {input_X:X_train, 
                                                                     input_y: y_train,keep_prob:1}))
                training_loss.append(s.run(loss, {input_X: X_train, 
                                                  input_y: y_train,keep_prob:1}))
            
                ## Evaluation of model
                validation_accuracy.append(accuracy_score(y_test.argmax(1), 
                                        s.run(predicted_y, {input_X: X_test,keep_prob:1}).argmax(1)))
                if verbose:
                    print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Val acc:{3:.3f}".format(epoch,
                                                                                    training_loss[epoch],
                                                                                    training_accuracy[epoch],
                                                                                    validation_accuracy[epoch]))
            
            
            #test_acc = accuracy_score(y_test.argmax(1),s.run(predicted_y, {input_X: X_test,keep_prob:1}).argmax(1))  
            testing_accuracy.append(validation_accuracy[-50:])   
            
            #print("Test acc:{0:.3f}".format(test_acc))   
            tf.reset_default_graph()
                        
        print("Final mean test accuracy: --- {0:.3f} ---)".format(np.mean(testing_accuracy)))            
        
        s.close()
            
        self.mlp_test_accuracy = testing_accuracy
        
        return np.mean(testing_accuracy)
        

    def univariate_top_features(self):
        """
        Compute the univariate classification accuracies
        """

        self.normalise_feature_data()

        X = self.X_norm
        y = self.y
        
        classification_accs = univariate_classification(X,y)
        
        self.univariate_classification_accuracy = classification_accs
        
    def pca_features_plot(self,X,y,indices): 
        """
        Compute the PCA of the feature set and plot it
        """

        from sklearn.decomposition import PCA
        import matplotlib.cm as cm  
        pca = PCA(n_components=2)
        
        X1 = X[np.argsort(y),:]
        y1 = y[np.argsort(y)]
        
        X_pca = pca.fit_transform(X1[:,indices]) 
        cm = cm.get_cmap('RdYlBu') 
        plt.scatter(X_pca[:,0],X_pca[:,1],cmap=cm,c=y1)                
        #plt.ylim([-30,30])
        #plt.xlim([-30,30])
        plt.xlabel('PC1')        
        plt.ylabel('PC2')
        
    def top_features_importance_plot(self,X,top_feat_indices,feature_names,y, name = 'xgboost', image_folder='images'):
        """ 
        Plot the top feature importances
        """

        import matplotlib.cm as cm  
        import random
        #mean_importance = np.mean(np.asarray(top_feats),0)                  
        #top_feat_indices = np.argsort(mean_importance)[::-1]  
        plt.figure()
        cm = cm.get_cmap('RdYlBu')                  
        sc = plt.scatter(X[:,top_feat_indices[0]],X[:,top_feat_indices[1]],cmap=cm,c=y)
        plt.xlabel(feature_names[top_feat_indices[0]])        
        plt.ylabel(feature_names[top_feat_indices[1]])
        plt.colorbar(sc)
        plt.savefig(image_folder + '/scatter_top2_feats_'+self.dataset+'_'+name+'.svg', bbox_inches = 'tight') 
        
    def plot_violin_feature(self,X,y,feature_id,feature_names, name = 'xgboost', image_folder='images'):   
        """
        Plot the violins of a feature
        """

        import random
        feature_data = X[:,feature_id]
        
        data_split = []
        for k in np.unique(y):
            indices = np.argwhere(y==k)
            data_split.append(feature_data[indices])        
        
        import seaborn as sns
        plt.figure()
        sns.set(style="whitegrid")
        ax = sns.violinplot(data=data_split,palette="muted",width=1)
        ax.set(xlabel='Class label', ylabel=feature_names[feature_id])
        plt.savefig(image_folder + '/violin_plot_'+self.dataset+'_'+feature_names[feature_id]+'_'+name+'.svg', bbox_inches = 'tight') 
        
    def save_feature_set(self,filename = 'Outputs/feature_set.pkl'):
        """
        Save the features in a pickle
        """

        import pickle as pkl        
        feature_set = self.graph_feature_set
        
        with open(filename,'wb') as output:
            pkl.dump(feature_set,output,pkl.HIGHEST_PROTOCOL)


    def load_feature_set(self,filename = 'Outputs/feature_set.pkl'):
        """
        Load the features from a pickle
        """

        import pickle as pkl
        with open(filename,'rb') as output:
            feature_set = pkl.load(output)
        
        self.graph_feature_set = feature_set
            
    def save_feature_matrix(self,filename = 'Outputs/feature_matrix.pkl'):
        """
        Save the features in a pickle
        """

        import pickle as pkl        
        feature_matrix = self.graph_feature_matrix
        
        with open(filename,'wb') as output:
            pkl.dump(feature_matrix,output,pkl.HIGHEST_PROTOCOL) 
        
    
    def load_feature_matrix(self,filename = 'Outputs/feature_matrix.pkl'):
        """
        Load the features from a pickle
        """

        import pickle as pkl
        with open(filename,'rb') as output:
            feature_matrix = pkl.load(output)
        
        self.graph_feature_matrix = feature_matrix


def classification(X,y,ml_model, verbose=True):
    """
    Perform classification of a normalized feature data
    """

    from sklearn.model_selection import StratifiedKFold  
    from sklearn.metrics import accuracy_score
    
    # reducing number of folds to half the least populated class
    # e.g. if only 9 elements of class A then we only use int(9/2)=4 folds
    counts = np.bincount(y)
    n_splits = int(np.min(counts[counts>0])/2)

    if n_splits < 2:
        n_splits = 2
        if verbose:
            print('Small dataset, we only do ', n_splits, ' splits.')

    elif n_splits > 10:
        n_splits = 10
    
    skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
         
    testing_accuracy = []
    
    if ml_model =='random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100,max_depth=30)
    elif ml_model == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(max_depth=4)
        
        
    top_feats = []
    
    for train_index, test_index in skf.split(X, y):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_pred = model.fit(X_train,y_train).predict(X_test)
        
        acc = accuracy_score(y_test,y_pred)
        
        if verbose:
            print("Fold test accuracy: --- {0:.3f} ---)".format(acc))            

        testing_accuracy.append(acc)         
        
        top_feats.append(model.feature_importances_)

    return testing_accuracy, top_feats
    


def calculate_features_single_graph(calc_speed, G):
    """
    Calculate the feature of a single graph, for parallel computations
    """

    G_operations = Operations(G)
    G_operations.feature_extraction(calc_speed=calc_speed)
    
    return G_operations


def univariate_classification(X,y):
    """
    Apply a univariate classification on each feature
    """
    
    classification_acc = []
    for i in range(X.shape[1]):
        testing_accuracy, top_feats = classification(X[:,i].reshape(-1,1),y,'xgboost',verbose=False)
        classification_acc.append(np.mean(testing_accuracy))
        
    return classification_acc
    

def top_features(X,top_feats,feature_names, image_folder, threshold = 0.9, plot=True):
    """
    Select and plot the dendogram, heatmap and importance distribution of top features
    """

    import pandas as pd
    from scipy.cluster.hierarchy import dendrogram, linkage
    import seaborn as sns
    
    mean_importance = np.mean(np.asarray(top_feats),0)                  
    sorted_mean_importance = np.sort(mean_importance)[::-1]     

    top_feat_indices = np.argsort(mean_importance)[::-1]            
    top_features_list = []
    for i in range(len(top_feat_indices)):                
        top_features_list.append(feature_names[top_feat_indices[i]])    
        
    top_features_list=list(dict.fromkeys(top_features_list))    
    top_features_list[:40]
    df_top40 = pd.DataFrame(columns = top_features_list[:40], data=X[:,top_feat_indices[:40]])            
    cor = np.abs(df_top40.corr())        
    Z = linkage(cor, 'ward')   
    
    if plot:
        plt.figure()
        dn = dendrogram(Z)         
        plt.savefig(image_folder+'/endogram_top40_features.svg', bbox_inches = 'tight') 

        new_index = [int(i) for  i in dn['ivl']]
        top_feats_names = [top_features_list[i] for i in new_index]             
        df = df_top40[top_feats_names]
        cor2 = np.abs(df.corr())            
        plt.figure()
        sns.heatmap(cor2, linewidth=0.5)
        plt.savefig(image_folder+'/heatmap_top40_feature_dependencies.svg', bbox_inches = 'tight') 

        # Taking only features till we have reached 90% importance
        sum_importance = 0      
        final_index = 0 
        for i in range(len(sorted_mean_importance)):
            sum_importance = sum_importance + sorted_mean_importance[i]
            if sum_importance > threshold:
                final_index = i
                break
        if final_index < 3: #take top 2 if no features are selected
            final_index = 3


        plt.figure()
        plt.plot(np.sort(mean_importance)[::-1])

        plt.xlabel('Features')
        plt.ylabel('Feature Importance')
        plt.xscale('log')
        plt.yscale('symlog', nonposy='clip', linthreshy=0.001)    
        plt.axvline(x=final_index,color='r')
        plt.savefig(image_folder+'/feature_importance_distribution.svg', bbox_inches = 'tight') 
    
    #import pickle as pkl        
    #pkl.dump(np.sort(mean_importance)[::-1], open('importance_data/'+image_folder+'.pkl','wb'), pkl.HIGHEST_PROTOCOL)

    return top_features_list

def reduce_feature_set(X,top_feats, threshold=0.9):
    """
    Reduce the feature set


    Parameters
    ---------
    top_feats: list
        List of features to keep
        
    """

    mean_importance = np.mean(np.asarray(top_feats),0)   
    sorted_mean_importance = np.sort(mean_importance)[::-1]     
    
    
    # Taking only features till we have reached 90% importance
    sum_importance = 0      
    final_index = 0 
    for i in range(len(sorted_mean_importance)):
        sum_importance = sum_importance + sorted_mean_importance[i]
        if sum_importance > threshold:
            final_index = i
            break
    if final_index < 3: #take top 2 if no features are selected
        final_index = 3


    top_feat_indices = np.argsort(mean_importance)[::-1][:final_index]     
        
    X_reduced = X[:,top_feat_indices]
    
    return X_reduced, top_feat_indices
