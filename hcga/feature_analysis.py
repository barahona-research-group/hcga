"""function for analysis of graph features"""

import numpy as np

def normalise_feature_data(g):
    """
    Normalise the feature matrix using sklearn scaler to remove the mean and scale to unit variance

    Parameters
    ----------

    g: Graphs instance from hcga.graphs to be modified
    """

    from sklearn.preprocessing import StandardScaler
    
    graph_feature_matrix = g.graph_feature_matrix
    
    X=graph_feature_matrix.values
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)     
    
    g.X_norm = X_norm
    g.y=np.asarray(g.graph_labels)

def define_feature_set(g, data='all'):
    """
    Select subset of features from graphs depending on options
    Parameters
    ----------

    g: Graphs instance from hcga.graphs to be modified

    data: string
        the type of features to classify
            * 'all' : all the features calculated
            * 'feats' : features based on node features and node labels only
            * 'topology' : features based only on the graph topology features

    Returns
    -------
    feature_names: list
        List of feature names to use
    """

    feature_names=[col for col in g.graph_feature_matrix.columns]            

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
    return feature_names

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

def compute_top_features(X,top_feats,feature_names):
    """
    Sort the top features
    """
    
    mean_importance = np.mean(np.asarray(top_feats),0)                  
    #sorted_mean_importance = np.sort(mean_importance)[::-1]     

    top_feat_indices = np.argsort(mean_importance)[::-1]            
    top_features_list = []
    for i in range(len(top_feat_indices)):                
        top_features_list.append(feature_names[top_feat_indices[i]])    
        
    top_features_list=list(dict.fromkeys(top_features_list))    

    return top_features_list, top_feat_indices 

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

def univariate_classification(X,y):
    """
    Apply a univariate classification on each feature
    """
    
    classification_acc = []
    for i in range(X.shape[1]):
        testing_accuracy, top_feats = classification(X[:,i].reshape(-1,1),y,'xgboost',verbose=False)
        classification_acc.append(np.mean(testing_accuracy))
        
    return classification_acc
