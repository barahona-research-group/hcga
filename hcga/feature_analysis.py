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

