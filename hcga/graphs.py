import numpy as np
import networkx as nx

from hcga.utils import read_graphfile
from hcga.Operations.operations import Operations

from tqdm import tqdm

from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

class Graphs():

    """
        Takes a list of graphs
    """

    def __init__(self, graphs = [], graph_meta_data = [], node_meta_data = [], graph_class = []):
        self.graphs = graphs # A list of networkx graphs
        self.graph_labels = graph_class # A list of class IDs - A single class ID for each graph.

        self.graph_metadata = graph_meta_data # A list of vectors with additional feature data describing the graph
        self.node_metadata = node_meta_data # A list of arrays with additional feature data describing nodes on the graph

        if not graphs:
            self.load_graphs()




    def load_graphs(self, directory = 'TestData', dataname = 'ENZYMES'):


        graphs,graph_labels = read_graphfile(directory,dataname)

        # selected data for testing code
        selected_data = np.arange(0,300,10)
        graphs = [graphs[i] for i in list(selected_data)]
        graph_labels = [graph_labels[i] for i in list(selected_data)]



        self.graphs = graphs
        self.graph_labels = graph_labels


    def calculate_features(self):
        """
        Calculation the features for each graph in the set of graphs

        """

        graph_feature_set = []

        for G in tqdm(self.graphs):
            G_operations = Operations(G)
            G_operations.feature_extraction()
            graph_feature_set.append(G_operations)


        self.calculated_graph_features = graph_feature_set

    def top_features(self):

        return



    def organise_feature_data(self):

        X = []
        for graph in self.calculated_graph_features:
            feature_names, features = graph._extract_data()
            X.append(features)

        X = np.asarray(X)
        y = np.asarray(self.graph_labels)

        self.X = X
        self.y = y

        return



    def normalise_feature_data(self):

        X = self.X

        X_N = X / X.max(axis=0)

        X_N = X_N[:,~np.isnan(X_N).any(axis=0)] # removing features with nan

        self.X_N = X_N

        return



    def graph_classification(self):

        """
        Graph Classification

        """

        from sklearn import model_selection
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC

        self.organise_feature_data()
        self.normalise_feature_data()

        X = self.X_N
        y = self.y


        # prepare configuration for cross validation test harness
        seed = 7

        # prepare models
        models = []
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(gamma='auto')))

        results = []
        names = []
        scoring = 'accuracy'
        for name, model in models:
        	kfold = model_selection.KFold(n_splits=10, random_state=seed)
        	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        	results.append(cv_results)
        	names.append(name)
        	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        	print(msg)

        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        #plt.show()

        return
