import logging

from .analysis import analysis, predict_evaluation_set
from .extraction import extract
from .io import load_dataset, load_features, save_dataset, save_features, load_fitted_model, save_fitted_model

import pandas as pd

import os
from pathlib import Path


# pylint: disable-all

L = logging.getLogger(__name__)
L.setLevel(logging.DEBUG)


class Hcga:
    """hcga standard object class."""

    def __init__(self):
        """init function."""
        self.graphs=None
        self.prediction_graphs=[]
        self.features=None
        self.features_prediction_set=None
        
    def load_data(
        self,
        dataset="./datasets/ENZYMES.pkl",
        labels=None,
        dataset_name="custom_data",
        folder="./datasets",
        prediction_graphs=False,
    ):
        if not prediction_graphs:
            if isinstance(dataset, str):
                # load graphs from pickle
                self.graphs = load_dataset(dataset)
            elif isinstance(dataset, list):
                # take graphs as list input
                save_dataset(dataset, labels, dataset_name, folder=folder)
                self.graphs = load_dataset(dataset)
        elif prediction_graphs:
            if isinstance(dataset, str):
                # load graphs from pickle
                self.prediction_graphs = load_dataset(dataset)
            elif isinstance(dataset, list):
                # take graphs as list input
                save_dataset(dataset, labels, dataset_name, folder=folder)
                self.prediction_graphs = load_dataset(dataset)

    def generate_data(self, dataset_name="ENZYMES", folder="./datasets"):
        """generate benchmark data."""

        if dataset_name == "TESTDATA":
            print("--- Building test dataset and creating pickle ---")
            from .dataset_creation import make_test_dataset

            make_test_dataset(folder=folder)
        else:
            print("---Downloading and creating pickle for {}---".format(dataset_name))
            from .dataset_creation import make_benchmark_dataset

            make_benchmark_dataset(dataset_name=dataset_name, folder=folder)

        

    def extract(
        self,
        n_workers=1,
        mode="slow",
        norm=True,
        stats_level="advanced",
        runtimes=False,
        node_feat=True,
        timeout=30,
        connected=False,
        weighted=True,
        prediction_set=False,
    ):
        
        if not prediction_set:
            self.features, self.features_info = extract(
                    self.graphs,
                    n_workers=int(n_workers),
                    mode=mode,
                    normalize_features=norm,
                    statistics_level=stats_level,
                    with_runtimes=runtimes,
                    with_node_features=node_feat,
                    timeout=timeout,
                    connected=connected,
            )
        elif prediction_set:
            self.features_prediction_set, self.features_info_prediction_set = extract(
                    self.prediction_graphs,
                    n_workers=int(n_workers),
                    mode=mode,
                    normalize_features=norm,
                    statistics_level=stats_level,
                    with_runtimes=runtimes,
                    with_node_features=node_feat,
                    timeout=timeout,
                    connected=connected,
            )
        

    def save_features(self, feature_file="./results/features.pkl"):
        save_features(
            self.features, self.features_info, self.graphs, filename=feature_file,
        )

    def load_features(self, feature_file="./results/features.pkl"):
        self.features, self.features_info, self.graphs = load_features(
            filename=feature_file
        )

    def save_model(self, model_file="./results/model.pkl"):
        save_features(
            self.model, filename=model_file,
        )

    def load_model(self, model_file="./results/model.pkl"):
        self.model = load_model(
            filename=model_file
        )
       

    def combine_features(self):
        self.full_feature_set = pd.concat([self.features,self.features_prediction_set],axis=0)

    def predict_unlabelled_graphs(
        self,
        graph_removal=0.3,
        interpretability=1,
        analysis_type="classification",
        model="XG",
        ):
        
        """ function to predict unlabelled graphs given training set """
        if self.features is None:
            raise Exception('You must have a feature set for training the model')
        if self.features_prediction_set is None:
            raise Exception('You must have a feature set for evaluating - do not forget to extract features for your predction set')
            
        self.combine_features()
        predictions = predict_evaluation_set(
                self.full_feature_set,
                self.features_info,
                analysis_type="classification",
                graph_removal=0.3,
                interpretability=1,
                model="XG",      
        )
        return predictions
        

    def analyse_features(
        self,
        feature_file=None,
        results_folder="./results",
        graph_removal=0.3,
        interpretability=1,
        analysis_type="classification",
        model="XG",
        kfold=True,
        reduce_set=True,
        reduced_set_size=100,
        reduced_set_max_correlation=0.9,
        plot=True,
        max_feats_plot=20,
        max_feats_plot_dendrogram=100,
        n_repeats=1,
        n_splits=None,
        random_state=42,
        test_size=0.2,
        trained_model=None,
        save_model=False,
    ):
        
        if feature_file is not None:
            self.load_features(feature_file=feature_file)
        

        analysis(
                self.features,
                self.features_info,
                self.graphs,
                analysis_type=analysis_type,
                folder=results_folder,
                graph_removal=graph_removal,
                interpretability=interpretability,
                model=model,
                kfold=kfold,
                reduce_set=reduce_set,
                reduced_set_size=reduced_set_size,
                reduced_set_max_correlation=reduced_set_max_correlation,
                plot=plot,
                max_feats_plot=max_feats_plot,
                max_feats_plot_dendrogram=max_feats_plot_dendrogram,
                n_repeats=n_repeats,
                n_splits=n_splits,
                random_state=random_state,
                test_size=test_size,
                trained_model=trained_model,
                save_model=save_model,
            )
