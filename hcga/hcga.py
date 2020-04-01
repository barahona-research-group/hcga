import networkx as nx
import numpy as np
import pandas as pd

from .analysis import analysis
from .extraction import extract
from .io import (load_dataset, load_features, save_analysis, save_dataset,
                 save_features)


class Hcga:
    """hcga standard object class.

    Parameters
    ----------

    """

    def __init__(self):
        """init function"""

    def load_data(
        self,
        dataset="./datasets/ENZYMES.pkl",
        labels=None,
        dataset_name="custom_data",
        folder="./datasets",
    ):

        if isinstance(dataset, str):
            # load graphs from pickle
            self.graphs = load_dataset(dataset)
        elif isinstance(dataset, list):
            # take graphs as list input
            save_dataset(dataset, labels, dataset_name, folder=folder)
            self.graphs = load_dataset(dataset)

    def generate_data(self, dataset_name="ENZYMES", folder="./datasets"):
        """ generate benchmark data """

        if dataset_name == "TESTDATA":
            print("--- Building test dataset and creating pickle ---")
            from .dataset_creation import make_test_dataset

            make_test_dataset(folder=folder)
        else:
            print("---Downloading and creating pickle for {}---".format(dataset_name))
            from .dataset_creation import make_benchmark_dataset

            make_benchmark_dataset(dataset_name=dataset_name, folder=folder)

    def extract(
        self, n_workers=1, mode="slow", norm=False, stats_level="basic", runtimes=False
    ):

        self.features, self.features_info = extract(
            self.graphs,
            n_workers=int(n_workers),
            mode=mode,
            normalize_features=norm,
            statistics_level=stats_level,
            with_runtimes=runtimes,
        )

        self.save_features()

    def save_features(self, feature_file="./results/features.pkl"):

        save_features(
            self.features, self.features_info, filename=feature_file,
        )

    def load_features(self, feature_file="./results/features.pkl"):

        [self.features, self.features_info] = load_features(filename=feature_file)

    def analyse_features(
        self,
        feature_file="./results/features.pkl",
        output_folder="./results",
        output_filename="features_analysis",
        interpretability=1,
        shap=True,
        classifier="RF",
        kfold=True,
        plot=True,
    ):

        self.load_features(feature_file=feature_file)

        X, explainer, shap_values = analysis(
            self.features,
            self.features_info,
            filename=output_filename,
            interpretability=interpretability,
            folder=output_folder,
            shap=shap,
            classifier=classifier,
            kfold=kfold,
            plot=plot,
        )
