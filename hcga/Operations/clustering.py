
import pandas as pd
import numpy as np
import networkx as nx

class Clustering():
    """
    Clustering class
    """    
    
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute the various clustering measures.

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list : dict
           Dictionary of features related to node clustering.


        Notes
        -----
        Implementation of networkx code:
            `Networkx_clustering <https://networkx.github.io/documentation/stable/reference/algorithms/clustering.html>`_

        We followed the same structure as networkx for implementing clustering features.

        """
        


        G = self.G

        feature_list = {}
        
        
        if not nx.is_directed(G):
            # Calculating number of triangles
            feature_list['num_triangles']=np.asarray(list(nx.triangles(G).values())).mean()
                
            # graph transivity
            C = nx.transitivity(G)
            feature_list['transitivity'] = C
        else:
            feature_list['num_triangles'] = np.nan
            feature_list['transitivity'] = np.nan
        

        # Average clustering coefficient
        feature_list['clustering_mean']=nx.average_clustering(G)
        feature_list['clustering_std']=np.asarray(list(nx.clustering(G).values())).std()
        feature_list['clustering_median']=np.median(np.asarray(list(nx.clustering(G).values())))


        # generalised degree
        feature_list['square_clustering_mean']=np.asarray(list(nx.square_clustering(G).values())).mean()
        feature_list['square_clustering_std']=np.asarray(list(nx.square_clustering(G).values())).std()
        feature_list['square_clustering_median']=np.median(np.asarray(list(nx.square_clustering(G).values())))
        

        


        self.features = feature_list
