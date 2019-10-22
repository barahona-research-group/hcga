

import numpy as np
import networkx as nx


class SpectrumModularity():
    """
    Spectrum modularity class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute measures of the spectrum of the modularity matrix



        Parameters
        ----------
        G : graph
          A networkx graph




        Returns
        -------
        feature_list : dict
           Dictionary of features related to spectrum of the modularity matrix


        Notes
        -----
        Spectral modularity calculations using networkx:
            `Networkx_spectrum_modularity <https://networkx.github.io/documentation/stable/reference/generated/networkx.linalg.spectrum.modularity_spectrum.html>`_


        """
        
        
        G = self.G

        feature_list = {}         
        

        # laplacian spectrum
        eigenvals_M = np.real(nx.linalg.spectrum.modularity_spectrum(G))
        
        if len(eigenvals_M) < 10:
            eigenvals_M = np.concatenate((eigenvals_M,np.zeros(10-len(eigenvals_M))))        
            
        for i in range(10):               
            feature_list['M_eigvals_'+str(i)]=eigenvals_M[i]
        
        
        for i in range(10):
            for j in range(10):
                if i != j:
                    try:
                        feature_list['M_eigvals_ratio_'+str(i)+'_'+str(j)] = eigenvals_M[i]/eigenvals_M[j]
                    except:
                        feature_list['M_eigvals_ratio_'+str(i)+'_'+str(j)] = np.nan
                
        feature_list['M_eigvals_min'] = min(eigenvals_M)




        

        self.features = feature_list
