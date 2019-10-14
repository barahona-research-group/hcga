import numpy as np
import pandas as pd
import networkx as nx

from hcga.utils import read_graphfile
from hcga.Operations.operations import Operations

from tqdm import tqdm
import time

from multiprocessing import Pool
from functools import partial

from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

class Graphs():

    """
        Takes a list of graphs
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

        # node labels are within the graph file
        
        # Node features are stored in: G.node[0]['feat'] 
        # Node labels are stored in: G.node[0]['label']
        
        if dataset == 'ENZYMES' or dataset == 'DD' or dataset == 'COLLAB' or dataset == 'PROTEINS' or dataset == 'REDDIT-MULTI-12K' :
        
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
            from hcga.TestData.HELICENES.graph_construction import construct_helicene_graphs
            graphs,graph_labels = construct_helicene_graphs()

            
        self.graphs = graphs
        self.graph_labels = graph_labels


    
    def calculate_features(self,calc_speed='slow'):
        """
        Calculation the features for each graph in the set of graphs

        """

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

        """
        calculate_features_single_graphf = partial(calculate_features_single_graph, calc_speed)
        
        with Pool(processes = self.n_processes) as p_feat:  #initialise the parallel computation
            self.graph_feature_set = list(tqdm(p_feat.imap(calculate_features_single_graphf, self.graphs), total = len(self.graphs)))


        
        comp_times = []
        for op in self.graph_feature_set:
            comp_times.append(list(op.computational_times.values()))
            
        comp_times_mean = np.mean(np.array(comp_times), axis = 0)
        
        feature_names = self.graph_feature_set[0]._extract_data()[0]
        for i, fn in enumerate(list(self.graph_feature_set[0].computational_times.keys())):
            print('Computation time for feature: ' + str(fn) + ' is ' + str(np.round(comp_times_mean[i],3)) + ' seconds.')
            
            
        # Create graph feature matrix
        feature_vals_matrix = np.empty([len(self.graph_feature_set),3*len(feature_names)])  
        
        # Append features for each graph as rows   
        for i in range(0,len(self.graph_feature_set)):            

            N = self.graphs[i].number_of_nodes()
            E = self.graphs[i].number_of_edges()
            graph_feats = np.asarray(self.graph_feature_set[i]._extract_data()[1])
            compounded_feats = np.hstack([graph_feats, graph_feats/N,graph_feats/E]) 
            
            
            
            feature_vals_matrix[i,:] = compounded_feats
        
        compounded_feature_names = feature_names + [s +'_N' for s in feature_names] + [s +'_E' for s in feature_names]
        
        raw_feature_matrix = pd.DataFrame(feature_vals_matrix, columns = compounded_feature_names)
        
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
        






    def normalise_feature_data(self):
        from sklearn.preprocessing import StandardScaler
        
        graph_feature_matrix = self.graph_feature_matrix
        
        X=graph_feature_matrix.values
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        
        self.X_norm = X_norm
        self.y=np.asarray(self.graph_labels)

        return



    def graph_classification(self,plot=True,ml_model='xgboost', data='all'):

        """
        Graph Classification
        
        data: the type of features to classify
            'all' - all the features calculated
            'feats' - features based on node features and node labels only
            'topology' - features based only on the graph topology features

        """
        


        
        """self.organise_feature_data()"""
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
        top_features_list = top_features(X,top_feats,feature_names)  
        self.top_feats = top_feats         
        self.top_features_list=top_features_list       
            
           
        # re running classification with reduced feature set
        X_reduced = reduce_feature_set(X,top_feats)        
        testing_accuracy_reduced_set, top_feats_reduced_set = classification(X_reduced,y,ml_model) 
        print("Final mean test accuracy reduced feature set: --- {0:.3f} ---)".format(np.mean(testing_accuracy_reduced_set)))            
        
    
        self.top_features_importance_plot(X,top_feats,y)
    
        self.test_accuracy = testing_accuracy_reduced_set 
        return np.mean(testing_accuracy_reduced_set)
    
    
    
    def graph_regression(self,plot=True, data='all'):
        
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
        


        
    def pca_features_plot(self,X,y,indices): 
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
        
    def top_features_importance_plot(self,X,top_feats,y):        
        import matplotlib.cm as cm        
        mean_importance = np.mean(np.asarray(top_feats),0)                  
        top_feat_indices = np.argsort(mean_importance)[::-1]  
        cm = cm.get_cmap('RdYlBu')                  
        sc = plt.scatter(X[:,top_feat_indices[0]],X[:,top_feat_indices[1]],cmap=cm,c=y)
        plt.xlabel(self.top_features_list[0])        
        plt.ylabel(self.top_features_list[1])
        plt.colorbar(sc)
        plt.savefig('Images/scatter_top2_feats_'+self.dataset+'.png') 
        

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


def classification(X,y,ml_model):
    from sklearn.model_selection import StratifiedKFold  
    from sklearn.metrics import accuracy_score
    # prepare configuration for cross validation test harness
    skf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)        
    
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
        
        top_feats.append(model.feature_importances_)

    return testing_accuracy, top_feats
    


def calculate_features_single_graph(calc_speed, G):
        
        G_operations = Operations(G)
        G_operations.feature_extraction(calc_speed=calc_speed)
        
        return G_operations


def top_features(X,top_feats,feature_names):
    import pandas as pd
    from scipy.cluster.hierarchy import dendrogram, linkage
    import seaborn as sns
    
    mean_importance = np.mean(np.asarray(top_feats),0)                  
    top_feat_indices = np.argsort(mean_importance)[::-1]            
    top_features_list = []
    for i in range(len(top_feat_indices)):                
        top_features_list.append(feature_names[top_feat_indices[i]])    
        
    top_features_list=list(dict.fromkeys(top_features_list))    
    top_features_list[:40]
    df_top40 = pd.DataFrame(columns = top_features_list[:40], data=X[:,top_feat_indices[:40]])            
    cor = np.abs(df_top40.corr())        
    Z = linkage(cor, 'ward')            
    plt.figure()
    dn = dendrogram(Z)         
    plt.savefig('Images/dendogram_top40_features.eps') 
    
    new_index = [int(i) for  i in dn['ivl']]
    top_feats_names = [top_features_list[i] for i in new_index]             
    df = df_top40[top_feats_names]
    cor2 = np.abs(df.corr())            
    plt.figure()
    ax = sns.heatmap(cor2, linewidth=0.5)
    plt.savefig('Images/heatmap_top40_feature_dependencies.eps') 
    
    
    plt.figure()
    plt.plot(np.sort(mean_importance)[::-1])
    plt.xlabel('Features')
    plt.ylabel('Feature Importance')
    plt.savefig('Images/feature_importance_distribution.eps') 
    

    return top_features_list

def reduce_feature_set(X,top_feats):
    mean_importance = np.mean(np.asarray(top_feats),0)                  
    top_feat_indices = np.argsort(mean_importance)[::-1][0:250]     
        
    X_reduced = X[:,top_feat_indices]
    return X_reduced