
import networkx as nx
from hcga.Operations import utils
import numpy as np

class Eccentricity():
    """
    Eccentricity class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []
        
    def feature_extraction(self):
        """
        Compute eccentricity for each node
        
        The eccentricity of a node is the maximum distance of the node from 
        any other node in the graph.
        
        Parameters
        ----------
        G : graph
          A networkx graph

        args :
            arg[0] Number of bins for calculating pdf of chosen distribution for SSE calculation

        Returns
        -------
        feature_list :list
           List of features related to eccentricity.


        Notes
        -----
        Eccentricity using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/distance_measures.html        
        """
        
        # Defining the input arguments
        bins = [10,20,50]
        
        """
        # Defining featurenames
        feature_names = ['mean','std','max','min']
        """
        
        G = self.G
        feature_list = {}
        if not nx.is_directed(G) or (nx.is_directed(G) and nx.is_strongly_connected(G)):
            #Calculate the eccentricity of each node
            eccentricity = np.asarray(list(nx.eccentricity(G).values()))
            # Basic stats regarding the eccentricity distribution
            feature_list['mean'] = eccentricity.mean()
            feature_list['std'] = eccentricity.std()
            feature_list['max'] = eccentricity.max()
            feature_list['min'] = eccentricity.min()
            
            for i in range(len(bins)):
                """# Adding to feature names
                feature_names.append('opt_model_{}'.format(bins[i]))
                feature_names.append('powerlaw_a_{}'.format(bins[i]))
                feature_names.append('powerlaw_SSE_{}'.format(bins[i]))"""
                
                # Fitting the eccentricity distribution and finding the optimal
                # distribution according to SSE
                opt_mod,opt_mod_sse = utils.best_fit_distribution(eccentricity,bins=bins[i])
                feature_list['opt_model_{}'.format(bins[i])] = opt_mod
    
                # Fitting power law and finding 'a' and the SSE of fit.
                feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(eccentricity,bins=bins[i])[0][-2]# value 'a' in power law
                feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(eccentricity,bins=bins[i])[1] # value sse in power law
        else:
            feature_list['eccentricity_calculations']='not implemented for not strongly connected digraphs'

        """
        self.feature_names=feature_names
        """
        self.features = feature_list
       
