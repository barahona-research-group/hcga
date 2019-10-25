# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np
import csv
from pprint import pprint
from importlib import import_module
import os
import networkx as nx

from scipy.sparse import linalg
import scipy as sp

import time
from tqdm import tqdm

class Operations():

    """
        Class that extracts all time-series features in a chosen YAML file
    """

    def __init__(self, G, CSVfilename = 'operations.csv'):

        self.G = G
        self.operations_dict = []
        self.CSVfilename = CSVfilename
        self.pre_computations = []
        self.feature_names = []
        self.feature_vals = []
        self.feature_dict = {}


        # pre computed values
        self.eigenvalues = []
        self.eigenvectors = []


        """
        The fact that some nodes are not connected needs to be changed. We need to append the subgraphs and run feature extraction
        on each subgraph. Add features that relate to the extra subgraphs. or features indicatnig there are subgraphs.
        """
        
        if not nx.is_connected(G):  
            Gc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            mapping=dict(zip(Gc.nodes,range(0,len(Gc))))
            Gc = nx.relabel_nodes(Gc,mapping)                
            self.G_largest_subgraph = Gc            
        else:
            self.G_largest_subgraph = G
        
        
        


        # functins to run automatically
        if not self.operations_dict:
            self.load_csv()

        if not self.pre_computations:
            self.pre_compute()

    def load_csv(self):
        
        module_dir = os.path.dirname(os.path.abspath(__file__))
        CSV_file_path = os.path.join(module_dir, self.CSVfilename)


        with open(CSV_file_path, 'r', newline='') as csv_file:
            reader = csv.reader(line.replace('  ', ',') for line in csv_file)
            operations_dict = list(reader)
            operations_dict.pop(0)

        self.operations_dict = operations_dict


    def pre_compute(self):

        """
        This function pre-computes a set of calculations that are often reused
        by various functions. This saves time by not pre-computing the same data
        for multiple functions.

        """
        if not self.eigenvalues:
            self.precompute_eigenvectors()




    def feature_extraction(self,calc_speed='slow'):
        operations_dict = self.operations_dict

        feature_dict = {}

        # loop over the feature classes defined in the YAML file
        calculation_speeds = ['fast']
        if calc_speed == 'medium':
            calculation_speeds.append('medium')
        elif calc_speed == 'slow':
            calculation_speeds.append('medium')
            calculation_speeds.append('slow')
        
        
        ###############
        # First add features to do with subgraphs  
        ###############
       
        filename = 'connected_components'
        classname = 'ConnectedComponents'
        feature_class = getattr(import_module('hcga.Operations.'+filename), classname)           
        feature_obj = feature_class(self.G)
        feature_obj.feature_extraction()
        feature_dict['CComp'] = feature_obj.features
        
        ###############
        
        # now looping over operations dictionary to calculate features
        self.computational_times = {}
        for i in range(len(operations_dict)):
            
            operation = operations_dict[i]

            #Extract the filename and class name
            #main_params = ['filename','classname','shortname','keywords','calculation_speed','precomputed']
            filename = operation[1]
            classname = operation[2]
            symbolic_name = operation[3]
            keywords = operation[4]
            calculation_speed = operation[5]
            precomputed = operation[6]

            # skip calculation if its too slow
            if calculation_speed not in calculation_speeds:
                continue

            # Extracting all additional arguments if they exist
            params = []   
            for arg in range(7,len(operation)):
                params.append(operation[arg])
                
            
            #print("Running file {} [{}/{}].".format(filename, i, len(operations_dict)))
            
            # import the class from the file            
            feature_class = getattr(import_module('hcga.Operations.'+filename), classname)           



            """# import the class from the file
            feature_class = getattr(import_module('hcga.Operations.'+filename), classname)"""



            start_time = time.time()                    

            if precomputed=='True ':
                feature_obj = feature_class(self.G_largest_subgraph,(self.eigenvalues,self.eigenvectors))
            else:
                feature_obj = feature_class(self.G_largest_subgraph)

            if not params:
                feature_obj.feature_extraction()
            else:
                feature_obj.feature_extraction(params)
                
            self.computational_times[classname] = time.time() - start_time
            
            # print("Time to calculate feature class "+ classname +"("+ symbolic_name +") :" + "--- %s seconds ---" % round(time.time() - start_time,3))               
            
            """
            # Alter the feature feature_names
            f_names = feature_obj.feature_names
            f_names_updated = [symbolic_name + '_' + f_name for f_name in f_names]
            """
            """
            # appending the parameter list onto the feature name
            param_string = '_'.join(map(str, params))
            if params:
                f_names_updated_params = [f_name + '_' + param_string for f_name in f_names_updated]
                f_names_updated = f_names_updated_params"""

            """
            # Append the altered feature names and the feature list of values
            feature_names.append(list(feature_obj.features.keys()))
            feature_vals.append(list(feature_obj.features.values()))
            """
            
            # Store features as a dictionary of dictionaries
            feature_dict[symbolic_name] = feature_obj.features
        
        
        self.feature_dict = feature_dict
        self.feature_vals = []
        self.feature_names = []

    def _extract_data(self):
        
        feature_dict = self.feature_dict
        feature_vals = [] 
        feature_names = [] 
        
        # Seperate out names and values from dictionary of features into 
        # feature_names and feature_vals
        symbolic_names=list(feature_dict.keys())
        features=list(feature_dict.values())
        for k in range(len(feature_dict)):
            names=list(features[k].keys())
            values=list(features[k].values())
            for l in range(len(names)):
                feature_names.append(symbolic_names[k]+'_'+names[l])
                feature_vals.append(values[l])
        
        """
        features_flat = [item for sublist in features for item in sublist] # features as single list
        feature_names_flat = [item for sublist in feature_names for item in sublist] # feature names as single list
        """
        
        return feature_names, feature_vals
        

    def precompute_eigenvectors(self,weight=None, max_iter=None, tol=1e-5):

        try:
            M = nx.to_scipy_sparse_matrix(self.G_largest_subgraph, nodelist=list(self.G_largest_subgraph), weight=weight,
                                      dtype=float)
            if not nx.is_directed(self.G_largest_subgraph):
                eigenvalues, eigenvectors = linalg.eigsh(M.T, k = int(0.8*self.G_largest_subgraph.number_of_nodes()), which='LM',
                                              maxiter=max_iter, tol=tol, v0 = numpy.ones(len(M))) #v0 is to fix randomness
            else:
                eigenvalues, eigenvectors = linalg.eigs(M.T, k = int(0.8*self.G_largest_subgraph.number_of_nodes()), which='LR',
                                              maxiter=max_iter, tol=tol, v0 = numpy.ones(len(M)))

            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors

        except:
            self.eigenvalues = np.array([1,1])
            self.eigenvectors = np.array([[1,1],[1,1]])
            pass




        return

