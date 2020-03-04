# -*- coding: utf-8 -*-
# This file is part of hcga.
#
# Copyright (C) 2019,
# Robert Peach (r.peach13@imperial.ac.uk),
# Alexis Arnaudon (alexis.arnaudon@epfl.ch),
# https://github.com/ImperialCollegeLondon/hcga.git
#
# hcga is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# hcga is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with hcga.  If not, see <http://www.gnu.org/licenses/>.


# to remove warnings
import numpy

numpy.seterr(all="ignore")
import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import pickle as pickle
import networkx as nx

from hcga.io import read_graphfile
from hcga.Operations.operations import Operations
import hcga.feature_analysis as hcga_analysis
import hcga.plotting as hcga_plotting

from tqdm import tqdm
import time

from multiprocessing import Pool
from functools import partial

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

# fix RNGs for reproducibility
import random

random.seed(10)
np.random.seed(10)


class Graphs:

    """
        Main class of hcga, with all the construction analysis functions. 
        The constructor takes either a list of networkx graphs, or a directory with a list of graph, and the name of the dataset (if it exists). 

        Parameters
        ----------

        graphs: list 
            list of graphs
        directory: folder
            directory with graphs
        dataset: string
            name of the dataset to load the graphs

    """

    def __init__(
        self,
        graphs=None,
        graph_meta_data=[],
        node_meta_data=[],
        graph_class=[],
        directory="",
        dataset="synthetic",
    ):

        self.graphs = graphs  # A list of networkx graphs
        self.graph_labels = (
            graph_class  # A list of class IDs - A single class ID for each graph.
        )

        self.graph_metadata = graph_meta_data  # A list of vectors with additional feature data describing the graph
        self.node_metadata = node_meta_data  # A list of arrays with additional feature data describing nodes on the graph
        self.n_processes = 4
        self.dataset = dataset

        if not graphs:
            self.load_graphs(directory=directory, dataset=dataset)

    def load_graphs(self, directory="", dataset="synthetic"):
        """
        Function to load graphs from a directory. 

        Parameters
        ----------
        directory: string
            directory path with graphs
        dataset: string
            Name of the dataset, the following are allowed: 
                * ENZYMES: benchmark 
                * DD: benchmark 
                * COLLAB: benchmark
                * PROTEINS: benchmark 
                * REDDIT-MULTI-12K: benchmark
                * synthetic: contains several variants of synthetics 
                * HELICENES: Julia Schmidt's dataset
                * NEURONS: neurons dataset for different animals 
        
        """

        if (
            dataset == "ENZYMES"
            or dataset == "DD"
            or dataset == "COLLAB"
            or dataset == "PROTEINS"
            or dataset == "REDDIT-MULTI-12K"
            or dataset == "ENZYMES1"
        ):

            if dataset == "ENZYMES1":
                graphs, graph_labels = read_graphfile(directory, "ENZYMES")
            else:
                graphs, graph_labels = read_graphfile(directory, dataset)

            to_remove = []
            for i, G in enumerate(graphs):

                # hack to add weights on edges if not present
                for u, v in G.edges:
                    if len(G[u][v]) == 0:
                        G[u][v]["weight"] = 1.0

                """
                The fact that some nodes are not connected needs to be changed. We need to append the subgraphs and run feature extraction
                on each subgraph. Add features that relate to the extra subgraphs. or features indicating there are subgraphs.
                """
                #                if not nx.is_connected(G):
                #                    print('Graph '+str(i)+' is not connected. Taking largest subgraph and relabelling the nodes.')
                #                    Gc = max(nx.connected_component_subgraphs(G), key=len)
                #                    mapping=dict(zip(Gc.nodes,range(0,len(Gc))))
                #                    Gc = nx.relabel_nodes(Gc,mapping)
                #                    graphs[i] = Gc

                if len(graphs[i]) < 3:
                    to_remove.append(i)

            # removing graphs with less than 2 nodes
            graph_labels = [i for j, i in enumerate(graph_labels) if j not in to_remove]
            graphs = [i for j, i in enumerate(graphs) if j not in to_remove]

        elif dataset == "synthetic":
            from hcga.TestData.create_synthetic_data import synthetic_data

            graphs, graph_labels = synthetic_data()

        elif dataset == "synthetic_watts_strogatz":
            from hcga.TestData.create_synthetic_data import (
                synthetic_data_watts_strogatz,
            )

            graphs, graph_labels = synthetic_data_watts_strogatz(N=1000)

        elif dataset == "synthetic_powerlaw_cluster":
            from hcga.TestData.create_synthetic_data import (
                synthetic_data_powerlaw_cluster,
            )

            graphs, graph_labels = synthetic_data_powerlaw_cluster(N=1000)

        elif dataset == "synthetic_sbm":
            from hcga.TestData.create_synthetic_data import synthetic_data_sbm

            graphs, graph_labels = synthetic_data_sbm(N=1000)

        elif dataset == "HELICENES":
            graphs, graph_labels = pickle.load(
                open(directory + "/HELICENES/helicenes_for_hcga.pkl", "rb")
            )

        elif dataset == "NEURONS":
            graphs_full = pickle.load(
                open(directory + "/NEURONS/neurons_animals_for_hcga.pkl", "rb")
            )

            graphs = []
            graph_labels = []
            for i in range(len(graphs_full)):
                G = graphs_full[i][0]
                graphs.append(graphs_full[i][0])
                graph_labels.append(graphs_full[i][1])

        self.graphs = graphs
        self.graph_labels = graph_labels

    def calculate_features(self, calc_speed="slow", parallel=True):
        """
        Extract the features from each graph in the set of graphs

        Parameters
        ----------

        calc_speed: string
            set of features to consider (from the operations.csv file). Can take 'slow', 'medium' or 'fast'. 
        parallel: bool
            True to run with multiprocessing 

        """

        if parallel:
            calculate_features_single_graphf = partial(
                calculate_features_single_graph, calc_speed
            )

            with Pool(
                processes=self.n_processes
            ) as p_feat:  # initialise the parallel computation
                self.graph_feature_set = list(
                    tqdm(
                        p_feat.imap(calculate_features_single_graphf, self.graphs),
                        total=len(self.graphs),
                    )
                )

            comp_times = []
            for op in self.graph_feature_set:
                comp_times.append(list(op.computational_times.values()))
            comp_times_mean = np.mean(np.array(comp_times), axis=0)

            # print the computational time from fast to slow
            sort_id = np.argsort(comp_times_mean)
            fns = []
            for i, fn in enumerate(
                list(self.graph_feature_set[0].computational_times.keys())
            ):
                fns.append(fn)

            for i in sort_id:
                print(
                    "Computation time for feature: "
                    + str(fns[i])
                    + " is "
                    + str(np.round(comp_times_mean[i], 3))
                    + " seconds."
                )

        else:
            graph_feature_set = []
            cnt = 0
            for G in tqdm(self.graphs):
                print("-------------------------------------------------")

                print("----- Computing features for graph " + str(cnt) + " -----")
                start_time = time.time()
                print("-------------------------------------------------")

                G_operations = Operations(G)
                G_operations.feature_extraction(calc_speed=calc_speed)
                graph_feature_set.append(G_operations)
                print("-------------------------------------------------")
                print(
                    "Time to calculate all features for graph "
                    + str(cnt)
                    + ": --- %s seconds ---" % round(time.time() - start_time, 3)
                )
                cnt = cnt + 1
                self.graph_feature_set_temp = graph_feature_set
            self.graph_feature_set = graph_feature_set

        feature_names = self.graph_feature_set[0]._extract_data()[0]

        # Create graph feature matrix
        feature_vals_matrix = np.empty(
            [len(self.graph_feature_set), 3 * len(feature_names)]
        )

        # Append features for each graph as rows
        for i in range(len(self.graph_feature_set)):

            N = self.graphs[i].number_of_nodes()
            E = self.graphs[i].number_of_edges()
            graph_feats = np.asarray(self.graph_feature_set[i]._extract_data()[1])
            compounded_feats = np.hstack(
                [graph_feats, graph_feats / N, graph_feats / E]
            )

            debug_features = False
            if debug_features:
                namesf = self.graph_feature_set[i]._extract_data()[0]
                for ci, cf in enumerate(self.graph_feature_set[i]._extract_data()[1]):
                    print(namesf[ci], cf)

            feature_vals_matrix[i, :] = compounded_feats

        compounded_feature_names = (
            feature_names
            + [s + "_N" for s in feature_names]
            + [s + "_E" for s in feature_names]
        )

        raw_feature_matrix = pd.DataFrame(
            feature_vals_matrix, columns=compounded_feature_names
        )

        self.raw_feature_matrix = raw_feature_matrix
        print("Number of raw features: ", np.shape(raw_feature_matrix)[1])

        # remove infinite and nan columns
        feature_matrix_clean = raw_feature_matrix.replace(
            [np.inf, -np.inf], np.nan
        ).dropna(1, how="any")
        print(
            "Number of features without nans/infs: ", np.shape(feature_matrix_clean)[1]
        )

        # remove columns with all zeros
        feats_all_zeros = (feature_matrix_clean == 0).all(0)
        feature_matrix_clean = feature_matrix_clean.drop(
            columns=feats_all_zeros[feats_all_zeros].index
        )

        # remove features with constant values
        feature_matrix_clean = feature_matrix_clean.loc[
            :, (feature_matrix_clean != feature_matrix_clean.iloc[0]).any()
        ]

        self.graph_feature_matrix = feature_matrix_clean
        print("Final number of features extracted:", np.shape(feature_matrix_clean)[1])

    def extract_feature(self, n):
        """
        Extract a feature from the feature matrix

        Parameters
        ----------
        n: int or string
            Either the feature number of its name

        Returns
        -------
        feature: list
            Feature values across graphs

        """

        graph_feature_matrix = self.graph_feature_matrix

        # if n is an int, extract column corresponding to n
        if type(n) == int:
            return graph_feature_matrix.iloc[:, n]
        # if n is a feature name, extract column corresponding to that feature name
        else:
            return graph_feature_matrix[n]

    def graph_classification(
        self,
        plot=True,
        ml_model="xgboost",
        data="all",
        reduc_threshold=0.9,
        image_folder="images",
    ):

        """
        Graph classification
       
        Parameters
        ----------

        data: string
            the type of features to classify
                * 'all' : all the features calculated
                * 'feats' : features based on node features and node labels only
                * 'topology' : features based only on the graph topology features
        ml_model: string
            ML method to use, can be:
                * xgboost
                * random_forest

        """

        hcga_analysis.normalise_feature_data(self)

        X = self.X_norm
        y = self.y

        feature_names = hcga_analysis.define_feature_set(self, data)

        testing_accuracy, top_feats = hcga_analysis.classification(X, y, ml_model)
        print(
            "Mean test accuracy full set: --- {0:.3f} ---)".format(
                np.mean(testing_accuracy)
            )
        )

        # get names of top features
        top_features_list, top_feat_indices = hcga_analysis.compute_top_features(
            X, top_feats, feature_names
        )

        self.top_feats = top_feats
        self.top_features_list = top_features_list

        if plot:
            hcga_plotting.plot_top_features(
                X,
                top_feats,
                feature_names,
                image_folder=image_folder,
                threshold=reduc_threshold,
            )
        # top_features(X,top_feats, feature_names, image_folder = image_folder, threshold = reduc_threshold, plot=plot)

        # re running classification with reduced feature set
        X_reduced, top_feat_indices = hcga_analysis.reduce_feature_set(
            X, top_feats, threshold=reduc_threshold
        )
        (
            testing_accuracy_reduced_set,
            top_feats_reduced_set,
        ) = hcga_analysis.classification(X_reduced, y, ml_model)
        print(
            "Final mean test accuracy reduced feature set: --- {0:.3f} ---)".format(
                np.mean(testing_accuracy_reduced_set)
            )
        )

        # top and violin plot top features
        if plot:
            hcga_plotting.top_features_importance_plot(
                self,
                X,
                top_feat_indices,
                feature_names,
                y,
                name="xgboost",
                image_folder=image_folder,
            )
            hcga_plotting.plot_violin_feature(
                self,
                X,
                y,
                top_feat_indices[0],
                feature_names,
                name="xgboost",
                image_folder=image_folder,
            )

        # univariate classification on top features
        univariate_topfeat_acc = hcga_analysis.univariate_classification(X_reduced, y)
        feature_names_reduced = [feature_names[i] for i in top_feat_indices]
        top_feat_index = np.argsort(univariate_topfeat_acc)[::-1]

        # top and violin plot top features from univariate
        if plot:
            hcga_plotting.top_features_importance_plot(
                self,
                X_reduced,
                top_feat_index[0:2],
                feature_names_reduced,
                y,
                name="univariate",
                image_folder=image_folder,
            )
            hcga_plotting.plot_violin_feature(
                self,
                X_reduced,
                y,
                top_feat_index[0],
                feature_names_reduced,
                name="univariate",
                image_folder=image_folder,
            )

        self.test_accuracy = testing_accuracy_reduced_set

        return np.mean(testing_accuracy_reduced_set)

    def save_feature_set(self, filename="Outputs/feature_set.pkl"):
        """
        Save the features in a pickle
        """

        import pickle as pkl

        feature_set = self.graph_feature_set

        with open(filename, "wb") as output:
            pkl.dump(feature_set, output, pkl.HIGHEST_PROTOCOL)

    def load_feature_set(self, filename="Outputs/feature_set.pkl"):
        """
        Load the features from a pickle
        """

        import pickle as pkl

        with open(filename, "rb") as output:
            feature_set = pkl.load(output)

        self.graph_feature_set = feature_set

    def save_feature_matrix(self, filename="Outputs/feature_matrix.pkl"):
        """
        Save the features in a pickle
        """

        import pickle as pkl

        feature_matrix = self.graph_feature_matrix

        with open(filename, "wb") as output:
            pkl.dump(feature_matrix, output, pkl.HIGHEST_PROTOCOL)

    def load_feature_matrix(self, filename="Outputs/feature_matrix.pkl"):
        """
        Load the features from a pickle
        """

        import pickle as pkl

        with open(filename, "rb") as output:
            feature_matrix = pkl.load(output)

        self.graph_feature_matrix = feature_matrix


def calculate_features_single_graph(calc_speed, G):
    """
    Calculate the feature of a single graph, for parallel computations
    """

    G_operations = Operations(G)
    G_operations.feature_extraction(calc_speed=calc_speed)

    return G_operations
