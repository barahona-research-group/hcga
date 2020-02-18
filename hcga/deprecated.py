"""
Deprecated functions no longer in use
"""

    def graph_regression(self,plot=True, data='all'):
        """
        Perform graph regression
        """ 

        from sklearn.model_selection import StratifiedKFold   
        import xgboost
        from sklearn.metrics import explained_variance_score

        hcga_analysis.normalise_feature_data(self)

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
        self.top_features_list = top_features_list            

            
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
        
        
        hcga_analysis.normalise_feature_data(self)

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

        hcga_analysis.normalise_feature_data(self)

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
        
