

import numpy as np
import networkx as nx


class SpectrumAdjacency():
    """
    Spectral adjacency class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute measures of the spectrum of the adjacency matrix



        Parameters
        ----------
        G : graph
          A networkx graph



        Returns
        -------
        feature_list : dict
           Dictionary of features related to spectrum of the adjacency matrix


        Notes
        -----
        Spectral adjacency calculations using networkx:
            `Networkx_spectrum_adjacency <https://networkx.github.io/documentation/stable/reference/generated/networkx.linalg.spectrum.adjacency_spectrum.html#networkx.linalg.spectrum.adjacency_spectrum>`_


        """

        
        
        G = self.G

        feature_list = {}         
        
        
        
        
        
        # laplacian spectrum
        eigenvals_A = np.real(nx.linalg.spectrum.adjacency_spectrum(G))
        
        if len(eigenvals_A) < 10:
            eigenvals_A = np.concatenate((eigenvals_A,np.zeros(10-len(eigenvals_A))))        
            
        for i in range(10):               
            feature_list['A_eigvals_'+str(i)]=eigenvals_A[i]
        
        
        for i in range(10):
            for j in range(i):
                try:
                    if abs(eigenvals_A[j] ) < 1e-8:
                        feature_list['A_eigvals_ratio_'+str(i)+'_'+str(j)] = 0 
                    else:
                        feature_list['A_eigvals_ratio_'+str(i)+'_'+str(j)] = eigenvals_A[i]/eigenvals_A[j]
                except Exception as e:
                    print('Exception for spectrum_adjacency', e)
                    feature_list['A_eigvals_ratio_'+str(i)+'_'+str(j)] = np.nan
                
        feature_list['A_eigvals_min'] = min(eigenvals_A)
        


        

        self.features = feature_list
