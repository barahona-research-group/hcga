from .analysis import analysis
from .extraction import extract
from .io import load_dataset, load_features, save_dataset, save_features

# pylint: disable-all


class Hcga:
    """hcga standard object class."""

    def __init__(self):
        """init function."""

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
        norm=False,
        stats_level="basic",
        runtimes=False,
        ensure_connectivity=True,
        node_feat=True,
    ):

        features, features_info = extract(
            self.graphs,
            n_workers=int(n_workers),
            mode=mode,
            normalize_features=norm,
            statistics_level=stats_level,
            with_runtimes=runtimes,
            with_node_features=node_feat,
            ensure_connectivity=ensure_connectivity,
        )

        self.save_features()

    def save_features(self, feature_file="./results/features.pkl"):

        save_features(
            self.features, self.features_info, filename=feature_file,
        )

    def load_features(self, feature_file="./results/features.pkl"):

        [self.features, self.features_info, self.graphs] = load_features(
            filename=feature_file
        )

    def analyse_features(
        self,
        feature_file="./results/features.pkl",
        results_folder="./results",
        interpretability=1,
        shap=True,
        grid_search=False,
        classifier="RF",
        kfold=True,
        plot=True,
    ):

        self.load_features(feature_file=feature_file)

        X, shap_values = analysis(
            self.features,
            self.features_info,
            self.graphs,
            interpretability=interpretability,
            grid_search=grid_search,
            folder=results_folder,
            shap=shap,
            classifier=classifier,
            kfold=kfold,
            plot=plot,
        )
