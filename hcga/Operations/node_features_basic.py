
import networkx as nx
import numpy as np
from hcga.Operations import utils


class NodeFeaturesBasic():
    """
    Node features class
    """
    
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}
        
    def feature_extraction(self):
        """
        Computes measures based on the node features/attributes
        


        Parameters
        ----------
        G : graph
           A networkx graph


        Returns
        -------
        feature_list : dict
           Dictionary of features related to 


        Notes
        -----

        Summary statistics for the node features. 
    


        """        
        G = self.G
        
        # Define number of bins
        bins = [10]
        
        feature_list = {}
        
        N = G.number_of_nodes()
        
        # Calculate degree of each node
        node_degrees = list(dict(nx.degree(G)).values())
        
        # Calculate some basic stats for degrees
        feature_list = utils.summary_statistics(feature_list,node_degrees,'node_degrees')
        
        # Distribution calculations and fit
        for i in range(len(bins)):        
                opt_mod,opt_mod_sse = utils.best_fit_distribution(node_degrees,bins=bins[i])
                feature_list['deg_opt_model_{}'.format(bins[i])] = opt_mod
                feature_list['deg_powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(node_degrees,bins=bins[i])[0][-2]# value 'a' in power law
                feature_list['deg_powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(node_degrees,bins=bins[i])[1] # value sse in power law
        
        # Only compute for networks with node features
        if 'feat' in G.nodes[0].keys():
            try:
                
                # Create a matrix features from each node
                node_matrix = np.array([G.nodes[0]['feat']])
                for i in range(1,N):
                    node_matrix = np.vstack([node_matrix,G.nodes[i]['feat']])
                
                
                num_feats = node_matrix.shape[1]
                
                for i in range(0,num_feats):                
                    feature_list['mean_feat'+str(i)] = np.mean(node_matrix,0)[i]
                    feature_list['max_feat'+str(i)] = np.max(node_matrix,0)[i]
                    feature_list['min_feat'+str(i)] = np.min(node_matrix,0)[i]
                    feature_list['median_feat'+str(i)] = np.median(node_matrix,0)[i]
                    feature_list['std_feat'+str(i)] = np.std(node_matrix,0)[i]
                    feature_list['sum_feat'+str(i)] = np.sum(node_matrix,0)[i]            
                
                # Calculate some basic stats from this matrix
                feature_list['mean'] = np.mean(node_matrix)
                feature_list['max'] = np.max(node_matrix)
                feature_list['min'] = np.min(node_matrix)
                feature_list['median'] = np.median(node_matrix)
                feature_list['std'] = np.std(node_matrix)
                feature_list['sum'] = np.sum(node_matrix)
                
                dim=np.shape(node_matrix)
                 
                # List containing the mean of each feature for all nodes
                mean_feat_val_list = [np.mean(node_matrix[:,i]) for i in range(dim[1])]
                
                # Calculate some basic stats from the mean of each feature
                feature_list['feat_mean_max'] = np.max(mean_feat_val_list)
                feature_list['feat_mean_min'] = np.min(mean_feat_val_list)
                feature_list['feat_mean_median'] = np.median(mean_feat_val_list)
                feature_list['feat_mean_std'] = np.std(mean_feat_val_list)
                
                # Distribution calculations and fit
                for i in range(len(bins)):        
                    opt_mod,opt_mod_sse = utils.best_fit_distribution(mean_feat_val_list,bins=bins[i])
                    feature_list['feat_opt_model_{}'.format(bins[i])] = opt_mod
                    feature_list['feat_powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(mean_feat_val_list,bins=bins[i])[0][-2]# value 'a' in power law
                    feature_list['feat_powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(mean_feat_val_list,bins=bins[i])[1] # value sse in power law
           
                # List containing the mean feature value for each node
                mean_node_feat_list = [np.mean(node_matrix[i,:]) for i in range(dim[0])]
                
                # Calculate some basic stats from the mean feature value for each node
                feature_list['node_mean_max'] = np.max(mean_node_feat_list)
                feature_list['node_mean_min'] = np.min(mean_node_feat_list)
                feature_list['node_mean_median'] = np.median(mean_node_feat_list)
                feature_list['node_mean_std'] = np.std(mean_node_feat_list)
                
                # Distribution calculations and fit
                for i in range(len(bins)):        
                    opt_mod,opt_mod_sse = utils.best_fit_distribution(mean_node_feat_list,bins=bins[i])
                    feature_list['node_opt_model_{}'.format(bins[i])] = opt_mod
                    feature_list['node_powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(mean_node_feat_list,bins=bins[i])[0][-2]# value 'a' in power law
                    feature_list['node_powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(mean_node_feat_list,bins=bins[i])[1] # value sse in power law
                
                # Divide the mean of the features of a node by its degree
                mean_node_feat_norm = [mean_node_feat_list[i]-node_degrees[i] for i in range(N)]
                
                # Calculate some basic stats for this normalisation
                feature_list['norm_mean'] = np.mean(mean_node_feat_norm)
                feature_list['norm_max'] = np.max(mean_node_feat_norm)
                feature_list['norm_min'] = np.min(mean_node_feat_norm)
                feature_list['norm_median'] = np.median(mean_node_feat_norm)
                feature_list['norm_std'] = np.std(mean_node_feat_norm)
                feature_list['norm_sum'] = np.sum(mean_node_feat_norm)
                
                # Distribution calculations and fit
                for i in range(len(bins)):        
                    opt_mod,opt_mod_sse = utils.best_fit_distribution(mean_node_feat_norm,bins=bins[i])
                    feature_list['norm_opt_model_{}'.format(bins[i])] = opt_mod
                    feature_list['norm_powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(mean_node_feat_norm,bins=bins[i])[0][-2]# value 'a' in power law
                    feature_list['norm_powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(mean_node_feat_norm,bins=bins[i])[1] # value sse in power law
                    
            except Exception as e:
                print('Exception for node_features_basic', e)

                num_feats = 0
                
                for i in range(0,num_feats):                
                    feature_list['mean_feat'+str(i)] = np.nan
                    feature_list['max_feat'+str(i)] = np.nan
                    feature_list['min_feat'+str(i)] = np.nan
                    feature_list['median_feat'+str(i)] = np.nan
                    feature_list['std_feat'+str(i)] = np.nan
                    feature_list['sum_feat'+str(i)] = np.nan       
                
                feature_list['mean'] = np.nan
                feature_list['max'] = np.nan
                feature_list['min'] = np.nan
                feature_list['median'] = np.nan
                feature_list['std'] = np.nan
                feature_list['sum'] = np.nan
                
                feature_list['feat_mean_max'] = np.nan
                feature_list['feat_mean_min'] = np.nan
                feature_list['feat_mean_median'] = np.nan
                feature_list['feat_mean_std'] = np.nan
                
                for i in range(len(bins)):        
                    feature_list['feat_opt_model_{}'.format(bins[i])] = np.nan
                    feature_list['feat_powerlaw_a_{}'.format(bins[i])] = np.nan
                    feature_list['feat_powerlaw_SSE_{}'.format(bins[i])] = np.nan
                    
                feature_list['node_mean_max'] = np.nan
                feature_list['node_mean_min'] = np.nan
                feature_list['node_mean_median'] = np.nan
                feature_list['node_mean_std'] = np.nan
                
                for i in range(len(bins)):        
                    feature_list['node_opt_model_{}'.format(bins[i])] = np.nan
                    feature_list['node_powerlaw_a_{}'.format(bins[i])] = np.nan
                    feature_list['node_powerlaw_SSE_{}'.format(bins[i])] = np.nan
                
                feature_list['norm_mean'] = np.nan
                feature_list['norm_max'] = np.nan
                feature_list['norm_min'] = np.nan
                feature_list['norm_median'] = np.nan
                feature_list['norm_std'] = np.nan
                
                for i in range(len(bins)):        
                    feature_list['norm_opt_model_{}'.format(bins[i])] = np.nan
                    feature_list['norm_powerlaw_a_{}'.format(bins[i])] = np.nan
                    feature_list['norm_powerlaw_SSE_{}'.format(bins[i])] = np.nan
                

        self.features=feature_list
