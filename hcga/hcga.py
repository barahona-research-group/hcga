"""Object-oriented API to hcga."""

import logging

import pandas as pd

from hcga.analysis import analysis, classify_pairwise
from hcga.extraction import extract
from hcga.io import load_dataset, load_features, load_fitted_model, save_features

L = logging.getLogger(__name__)
L.setLevel(logging.DEBUG)


class Hcga:
    """hcga standard object class."""

    def __init__(self):
        """init function."""
        self.graphs = None
        self.features = None
        self.features_info = None

    def load_data(
        self,
        dataset="./datasets/ENZYMES.pkl",
        # labels=None,
        # folder="./datasets",
        prediction_graphs=False,
    ):
        """Load dataset."""
        if not prediction_graphs:
            if isinstance(dataset, str):
                # load graphs from pickle
                self.graphs = load_dataset(dataset)
            elif isinstance(dataset, list):
                # take graphs as list input
                # save_dataset(dataset, labels, dataset_name, folder=folder)
                # self.graphs = load_dataset(dataset)
                raise Exception("Not implemented")
        elif prediction_graphs:
            if isinstance(dataset, str):
                # load graphs from pickle
                self.prediction_graphs = load_dataset(dataset)
            elif isinstance(dataset, list):
                # take graphs as list input
                # save_dataset(dataset, labels, dataset_name, folder=folder)
                #  self.prediction_graphs = load_dataset(dataset)
                raise Exception("Not implemented")

    def generate_data(self, dataset_name="ENZYMES", folder="./datasets"):
        """generate benchmark data."""

        if dataset_name == "TESTDATA":
            print("--- Building test dataset and creating pickle ---")
            from .dataset_creation import make_test_dataset

            make_test_dataset(folder=folder)
        else:
            print(f"---Downloading and creating pickle for {dataset_name}---")
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
        prediction_set=False,
    ):
        """Exctract features."""

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
        """Save features."""
        save_features(
            self.features,
            self.features_info,
            self.graphs,
            filename=feature_file,
        )

    def load_features(self, feature_file="./results/features.pkl"):
        """Load features."""
        self.features, self.features_info, self.graphs = load_features(filename=feature_file)

    # def save_model(self, model_file="./results/model.pkl"):
    #    """Save model."""
    #    save_fitted_model(
    #        self.model,
    #        filename=model_file,
    #    )

    def load_model(self, model_file="./results/model.pkl"):
        """Load model."""
        self.model = load_fitted_model(filename=model_file)

    def combine_features(self):
        """Combine features."""
        self.full_feature_set = pd.concat([self.features, self.features_prediction_set], axis=0)

    def analyse_features(
        self,
        feature_file=None,
        results_folder="./results",
        graph_removal=0.3,
        interpretability=1,
        analysis_type="classification",
        model="XG",
        compute_shap=True,
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
        """Analyse features."""

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
            compute_shap=compute_shap,
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

    def pairwise_classification(
        self,
        feature_file=None,
        model="XG",
        graph_removal=0.3,
        interpretability=1,
        n_top_features=5,
        reduce_set=False,
        reduced_set_size=100,
        reduced_set_max_correlation=0.5,
        n_repeats=1,
        n_splits=None,
        analysis_type="classification",
    ):
        """Apply pairwise classification."""

        if feature_file is None:
            try:
                features = self.features
                features_info = self.features_info
            except Exception as e:  # pylint: disable=broad-except
                raise Exception("You have not loaded any feature data.") from e

        if isinstance(feature_file, str):
            self.load_features(feature_file=feature_file)
            features = self.features
            features_info = self.features_info

        accuracy_matrix, top_features = classify_pairwise(
            features,
            features_info,
            model=model,
            graph_removal=graph_removal,
            interpretability=interpretability,
            n_top_features=n_top_features,
            reduce_set=reduce_set,
            reduced_set_size=reduced_set_size,
            reduced_set_max_correlation=reduced_set_max_correlation,
            n_repeats=n_repeats,
            n_splits=n_splits,
            analysis_type=analysis_type,
        )

        return accuracy_matrix, top_features
