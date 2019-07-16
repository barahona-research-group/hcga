
from networkx.algorithms import assortativity
import numpy as np

class AverageNeighborDegree():
    
    
    
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []
        
    def feature_extraction(self):
        
        feature_names = ['mean','std']
        G = self.G
        feature_list = []
        average_neighbor_degree = np.asarray(list(assortativity.average_neighbor_degree(G).values()))
        feature_list.append(average_neighbor_degree.mean())
        feature_list.append(average_neighbor_degree.std())
        
        self.feature_names=feature_names
        self.features = feature_list
        
