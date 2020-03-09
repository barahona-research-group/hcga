""" hcga app with click module """
import click

import numpy
import warnings
from pathlib import Path

numpy.seterr(all="ignore")
warnings.simplefilter("ignore")


@click.group()
def cli():
    """init app"""


@cli.command("extract_features")
@click.argument("dataset", type=str)
@click.option(
    "-n", "--n-workers", default=1, help="Number of workers for multiprocessing"
)
@click.option("-m", "--mode", default="fast", help="Mode of features to extract")
@click.option(
    "--norm/--no-norm",
    default=False,
    help="Normalised features by number of edges/nodes",
)
@click.option(
    "-sl",
    "--stats-level",
    default="basic",
    help="Level of statistical features (basic, medium, advanced)",
)
@click.option("-of", "--output-file", help="Location of results")
@click.option("--runtimes/--no-runtimes", default=False, help="output runtimes")
def extract_features(
    dataset,
    n_workers,
    mode,
    output_file,
    norm,
    stats_level,
    runtimes,
):
    """Extract features from dataset of graphs and save the feature matrix, info and labels"""
    from .io import load_dataset, save_features
    from .feature_extraction import extract

    graphs = load_dataset(dataset)

    features, features_info = extract(
        graphs,
        n_workers=int(n_workers),
        mode=mode,
        normalize_features=norm,
        statistics_level=stats_level,
        with_runtimes=runtimes,
    )

    if output_file is None:
        output_file = Path(dataset).parent / (Path(dataset).stem + '_features.pkl')

    save_features(
        features, features_info, filename=output_file,
    )


@cli.command("feature_analysis")
@click.argument(
    "feature_file", type=str
)
@click.option(
    "-ff", "--feature-folder", default="./results", help="Location of results"
)
@click.option("-m", "--mode", default="sklearn", help="mode of feature analysis")
@click.option("-c", "--classifier", default="RF", help="classifier feature analysis")
@click.option("--kfold/--no-kfold", default=False, help="use K-fold")
def feature_analysis(feature_file, feature_folder, mode, classifier, kfold):
    """Extract features from dataset of graphs"""
    from .io import load_features, save_analysis
    from .feature_analysis import analysis

    features, features_info = load_features(filename=feature_file)
    X, testing_accuracy, top_features = analysis(
        features,
        features_info,
        folder=feature_folder,
        mode=mode,
        classifier_type=classifier,
        kfold=kfold,
    )
    save_analysis(X, testing_accuracy, top_features, folder=feature_folder)


@cli.command("plot_analysis")
@click.option(
    "-ff", "--feature-folder", default="./results", help="Location of results"
)
def plot_analysis(feature_folder):
    """Extract features from dataset of graphs"""
    from .io import load_analysis
    from .plotting import plot_sklearn_analysis

    X, testing_accuracy, top_features = load_analysis(folder=feature_folder)
    plot_sklearn_analysis(X, testing_accuracy, top_features, folder=feature_folder)


@cli.command("get_data")
@click.argument("dataset_name", type=str)
@click.option("-f", "--folder", default="./datasets", help="Location to save dataset")
def generate_data(dataset_name, folder):
    """Generate the benchmark or test data"""
    if dataset_name == "TESTDATA":
        print("--- Building test dataset and creating pickle ---")
        from .dataset_creation import make_test_dataset

        make_test_dataset(folder=folder)
    else:
        print("---Downloading and creating pickle for {}---".format(dataset_name))
        from .dataset_creation import make_benchmark_dataset

        make_benchmark_dataset(dataset_name=dataset_name, folder=folder)
