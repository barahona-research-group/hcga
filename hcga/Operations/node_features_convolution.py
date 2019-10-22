
import networkx as nx
import numpy as np
from hcga.Operations import utils


class NodeFeaturesConv():
    """
    Node features convoluted class    
    """
    
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}
        
    def feature_extraction(self):
        """
        Computes measures based on the node features/attributes convoluted over the graph structure.
        We implement a message passing system using the discrete adjacency matrix.        


        Parameters
        ----------
        G : graph
           A networkx graph


        Returns
        -------
        feature_list : dict
           Dictionary of features related to node features convoluted


        Notes
        -----

        Summary statistics of convoluted node features.
    


        """                
        G = self.G
        
        # Define number of bins
        bins = [10]
        
        feature_list = {}
        
        N = G.number_of_nodes()
        
        node_degrees = list(dict(nx.degree(G)).values())


        try:
            
            # Create a matrix features from each node
            node_matrix = np.array([G.nodes[0]['feat']])
            for i in range(1,N):
                node_matrix = np.vstack([node_matrix,G.nodes[i]['feat']])
            
            
            num_feats = node_matrix.shape[1]
            
            A = nx.to_numpy_array(G)
            A = A + np.eye(len(G))
            
            
            for conv in range(2):
                
                # each loop imposes a one step random walk convolution
                node_matrix = np.dot(A,node_matrix)
                
                
                for i in range(0,num_feats):                
                    feature_list['mean_feat_conv'+str(conv)+'_'+str(i)] = np.mean(node_matrix,0)[i]
                    feature_list['max_feat_conv'+str(conv)+'_'+str(i)] = np.max(node_matrix,0)[i]
                    feature_list['min_feat_conv'+str(conv)+'_'+str(i)] = np.min(node_matrix,0)[i]
                    feature_list['median_feat_conv'+str(conv)+'_'+str(i)] = np.median(node_matrix,0)[i]
                    feature_list['std_feat_conv'+str(conv)+'_'+str(i)] = np.std(node_matrix,0)[i]
                    feature_list['sum_feat_conv'+str(conv)+'_'+str(i)] = np.sum(node_matrix,0)[i]      
    
                # Calculate some basic stats from this matrix
                feature_list['mean_conv'+str(conv)] = np.mean(node_matrix)
                feature_list['max_conv'+str(conv)] = np.max(node_matrix)
                feature_list['min_conv'+str(conv)] = np.min(node_matrix)
                feature_list['median_conv'+str(conv)] = np.median(node_matrix)
                feature_list['std_conv'+str(conv)] = np.std(node_matrix)
                feature_list['sum_conv'+str(conv)] = np.sum(node_matrix)
                
                dim=np.shape(node_matrix)
                 
                # List containing the mean of each feature for all nodes
                mean_feat_val_list = [np.mean(node_matrix[:,i]) for i in range(dim[1])]
                
                # Calculate some basic stats from the mean of each feature
                feature_list['feat_mean_max_conv'+str(conv)] = np.max(mean_feat_val_list)
                feature_list['feat_mean_min_conv'+str(conv)] = np.min(mean_feat_val_list)
                feature_list['feat_mean_median_conv'+str(conv)] = np.median(mean_feat_val_list)
                feature_list['feat_mean_std_conv'+str(conv)] = np.std(mean_feat_val_list)
                
                # Distribution calculations and fit
                for i in range(len(bins)):        
                    opt_mod,opt_mod_sse = utils.best_fit_distribution(mean_feat_val_list,bins=bins[i])
                    feature_list[str(conv)+'feat_opt_model_{}'.format(bins[i])] = opt_mod
                    feature_list[str(conv)+'feat_powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(mean_feat_val_list,bins=bins[i])[0][-2]# value 'a' in power law
                    feature_list[str(conv)+'feat_powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(mean_feat_val_list,bins=bins[i])[1] # value sse in power law
           
                # List containing the mean feature value for each node
                mean_node_feat_list = [np.mean(node_matrix[i,:]) for i in range(dim[0])]
                
                # Calculate some basic stats from the mean feature value for each node
                feature_list['node_mean_max_conv'+str(conv)] = np.max(mean_node_feat_list)
                feature_list['node_mean_min_conv'+str(conv)] = np.min(mean_node_feat_list)
                feature_list['node_mean_median_conv'+str(conv)] = np.median(mean_node_feat_list)
                feature_list['node_mean_std_conv'+str(conv)] = np.std(mean_node_feat_list)
                
                # Distribution calculations and fit
                for i in range(len(bins)):        
                    opt_mod,opt_mod_sse = utils.best_fit_distribution(mean_node_feat_list,bins=bins[i])
                    feature_list[str(conv)+'node_opt_model_{}'.format(bins[i])] = opt_mod
                    feature_list[str(conv)+'node_powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(mean_node_feat_list,bins=bins[i])[0][-2]# value 'a' in power law
                    feature_list[str(conv)+'node_powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(mean_node_feat_list,bins=bins[i])[1] # value sse in power law
                
                # Divide the mean of the features of a node by its degree
                mean_node_feat_norm = [mean_node_feat_list[i]/node_degrees[i] for i in range(N)]
                
                # Calculate some basic stats for this normalisation
                feature_list['norm_mean_conv'+str(conv)] = np.mean(mean_node_feat_norm)
                feature_list['norm_max_conv'+str(conv)] = np.max(mean_node_feat_norm)
                feature_list['norm_min_conv'+str(conv)] = np.min(mean_node_feat_norm)
                feature_list['norm_median_conv'+str(conv)] = np.median(mean_node_feat_norm)
                feature_list['norm_std_conv'+str(conv)] = np.std(mean_node_feat_norm)
                feature_list['norm_sum_conv'+str(conv)] = np.sum(mean_node_feat_norm)
                
                # Distribution calculations and fit
                for i in range(len(bins)):        
                    opt_mod,opt_mod_sse = utils.best_fit_distribution(mean_node_feat_norm,bins=bins[i])
                    feature_list[str(conv)+'norm_opt_model_{}'.format(bins[i])] = opt_mod
                    feature_list[str(conv)+'norm_powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(mean_node_feat_norm,bins=bins[i])[0][-2]# value 'a' in power law
                    feature_list[str(conv)+'norm_powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(mean_node_feat_norm,bins=bins[i])[1] # value sse in power law

            
        except Exception as e:
            print('Exception for node_features_conv:', e)

            for conv in range(2):
                
                # each loop imposes a one step random walk convolution
                num_feats = 0
                
                for i in range(0,num_feats):                
                    feature_list['mean_feat_conv'+str(conv)+'_'+str(i)] = np.nan
                    feature_list['max_feat_conv'+str(conv)+'_'+str(i)] = np.nan
                    feature_list['min_feat_conv'+str(conv)+'_'+str(i)] = np.nan
                    feature_list['median_feat_conv'+str(conv)+'_'+str(i)] = np.nan
                    feature_list['std_feat_conv'+str(conv)+'_'+str(i)] = np.nan
                    feature_list['sum_feat_conv'+str(conv)+'_'+str(i)] = np.nan   
    
                # Calculate some basic stats from this matrix
                # Calculate some basic stats from this matrix
                feature_list['mean_conv'+str(conv)] = np.nan
                feature_list['max_conv'+str(conv)] = np.nan
                feature_list['min_conv'+str(conv)] = np.nan
                feature_list['median_conv'+str(conv)] = np.nan
                feature_list['std_conv'+str(conv)] = np.nan
                feature_list['sum_conv'+str(conv)] = np.nan

                
                # Calculate some basic stats from the mean of each feature
                feature_list['feat_mean_max_conv'+str(conv)] = np.nan
                feature_list['feat_mean_min_conv'+str(conv)] = np.nan
                feature_list['feat_mean_median_conv'+str(conv)] = np.nan
                feature_list['feat_mean_std_conv'+str(conv)] = np.nan
                
                
                # Distribution calculations and fit
                for i in range(len(bins)):        
                    
                    feature_list[str(conv)+'feat_opt_model_{}'.format(bins[i])] = np.nan
                    feature_list[str(conv)+'feat_powerlaw_a_{}'.format(bins[i])] = np.nan
                    feature_list[str(conv)+'feat_powerlaw_SSE_{}'.format(bins[i])] = np.nan
           

                # Calculate some basic stats from the mean feature value for each node
                feature_list['node_mean_max_conv'+str(conv)] = np.nan
                feature_list['node_mean_min_conv'+str(conv)] = np.nan
                feature_list['node_mean_median_conv'+str(conv)] = np.nan
                feature_list['node_mean_std_conv'+str(conv)] = np.nan
                
                # Distribution calculations and fit
                for i in range(len(bins)):        

                    feature_list[str(conv)+'node_opt_model_{}'.format(bins[i])] = np.nan
                    feature_list[str(conv)+'node_powerlaw_a_{}'.format(bins[i])] = np.nan
                    feature_list[str(conv)+'node_powerlaw_SSE_{}'.format(bins[i])] = np.nan
                
                
                # Calculate some basic stats for this normalisation
                feature_list['norm_mean_conv'+str(conv)] = np.nan
                feature_list['norm_max_conv'+str(conv)] = np.nan
                feature_list['norm_min_conv'+str(conv)] = np.nan
                feature_list['norm_median_conv'+str(conv)] = np.nan
                feature_list['norm_std_conv'+str(conv)] = np.nan
                feature_list['norm_sum_conv'+str(conv)] = np.nan
                
                # Distribution calculations and fit
                for i in range(len(bins)):        

                    feature_list[str(conv)+'norm_opt_model_{}'.format(bins[i])] = np.nan
                    feature_list[str(conv)+'norm_powerlaw_a_{}'.format(bins[i])] = np.nan
                    feature_list[str(conv)+'norm_powerlaw_SSE_{}'.format(bins[i])] = np.nan



        self.features=feature_list
