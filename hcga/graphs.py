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
        
        if dataset == 'ENZYMES' or dataset == 'DD' :
        
            graphs,graph_labels = read_graphfile(directory,dataset)
    
            # selected data for testing code from the enzymes dataset...
            #selected_data = np.arange(0,600,1) 
            #graphs = [graphs[i] for i in list(selected_data)]
            #graph_labels = [graph_labels[i] for i in list(selected_data)]
            
            
            to_remove = []
            for i,G in enumerate(graphs): 
                """
                The fact that some nodes are not connected needs to be changed. We need to append the subgraphs and run feature extraction
                on each subgraph. Add features that relate to the extra subgraphs. or features indicatnig there are subgraphs.
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
            
        self.graphs = graphs
        self.graph_labels = graph_labels


    def calculate_features(self,calc_speed='slow'):
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
            G_operations.feature_extraction(calc_speed=calc_speed)
            graph_feature_set.append(G_operations)
            print("-------------------------------------------------")               
            print("Time to calculate all features for graph "+ str(cnt) +": --- %s seconds ---" % round(time.time() - start_time,3))               
            cnt = cnt+1
            self.graph_feature_set_temp = graph_feature_set
            
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
        self.save_feature_set()

        
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
        X=self.X_norm
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
        
        X=self.X_norm
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
            
        
    def top_features_leaveoneout(self):
        
        X = self.X_norm
        y = self.y
        
        feature_names=[col for col in self.graph_feature_matrix.columns]
        
        original_acc = self.graph_classification_mlp(verbose=False)
        
        accuracy = []
        
        for i in range(X.shape[1]):
            mask = np.ones(X.shape[1], dtype=bool)
            mask[i] = False
            X_leaveoneout = X[:,mask]        
            acc = self.graph_classification_mlp(X=X_leaveoneout,y=y,verbose=False)
            accuracy.append(acc)
            
        reduction_accuracy = original_acc - np.asarray(accuracy)   
        
        top_feat_ids = np.flip(np.argsort(reduction_accuracy))
        top_feat_acc = np.flip(np.sort(reduction_accuracy))
        top_feats_names = [feature_names[i] for i in top_feat_ids]        
        
        self.top_feat_leaveoneout = (top_feat_ids,top_feat_acc,top_feats_names)
        
        
    
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
        from sklearn.preprocessing import StandardScaler
        
        graph_feature_matrix = self.graph_feature_matrix
        
        X=graph_feature_matrix.values
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        
        self.X_norm = X_norm
        self.y=np.asarray(self.graph_labels)

        return



    def graph_classification(self,plot=True,ml_model='random_forest'):

        """
        Graph Classification

        """
        
        from sklearn.model_selection import StratifiedKFold  
        from sklearn.metrics import accuracy_score

        """self.organise_feature_data()"""
        self.normalise_feature_data()

        X = self.X_norm
        y = self.y


        # prepare configuration for cross validation test harness



        skf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
        testing_accuracy = []
        
        if ml_model =='random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100,max_depth=30)
        elif ml_model == 'xgboost':
            from xgboost import XGBClassifier
            model = XGBClassifier(max_depth=4)
            

        for train_index, test_index in skf.split(X, y):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #X_train, y_train, X_test, y_test = load_dataset(X,y)
            
            ## Changing labels to one-hot encoded vector
            #lb = LabelBinarizer()
            #y_train = lb.fit_transform(y_train)
            #y_test = lb.transform(y_test)
    
            print('Train labels dimension:');print(y_train.shape)
            print('Test labels dimension:');print(y_test.shape)   
            
            #rf = RandomForestClassifier(n_estimators=100,max_depth=100,max_features=None)
            y_pred = model.fit(X_train,y_train).predict(X_test)
            
            acc = accuracy_score(y_test,y_pred)
            print("Fold test accuracy: --- {0:.3f} ---)".format(acc))            

            testing_accuracy.append(acc)   
            
        
        print("Final mean test accuracy: --- {0:.3f} ---)".format(np.mean(testing_accuracy)))            
            
            
        self.test_accuracy = testing_accuracy
            

#        for name, model in models:
#            	kfold = model_selection.KFold(n_splits=10, random_state=10)
#            	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
#            	results.append(cv_results)
#            	names.append(name)
#            	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#            	print(msg)
#
#        if plot:
#            fig = plt.figure()
#            fig.suptitle('Algorithm Comparison')
#            ax = fig.add_subplot(111)
#            plt.boxplot(results)
#            plt.ylim(0,1)
#            ax.set_xticklabels(names)
        #plt.show()

        return np.mean(testing_accuracy)
    
    
    
    def graph_classification_mlp(self,X = None, y = None , verbose=True):
        
        #from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelBinarizer        
        from sklearn.model_selection import StratifiedKFold          
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import accuracy_score
               

        if X is None:
            X = self.X_norm
            y = self.y        
       
        testing_accuracy = []
        
        skf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
        
        
        for train_index, test_index in skf.split(X, y):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #X_train, y_train, X_test, y_test = load_dataset(X,y)
            
            ## Changing labels to one-hot encoded vector
            lb = LabelBinarizer()
            y_train = lb.fit_transform(y_train)
            y_test = lb.transform(y_test)
    
            print('Train labels dimension:');print(y_train.shape)
            print('Test labels dimension:');print(y_test.shape)            
            

            s = tf.Session()  # Create new session            
            
            ## Defining various initialization parameters for 600-256-128-3 MLP model
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
            
            
        self.mlp_test_accuracy = testing_accuracy
        
        return np.mean(testing_accuracy)
        

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

    