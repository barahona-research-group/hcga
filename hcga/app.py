""" hcga app with click module """
import click

import numpy
import warnings

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
    "-df", "--dataset-folder", default="./datasets", help="Location of dataset"
)
@click.option("-of", "--output-folder", default="./results", help="Location of results")
@click.option("-on", "--output-name", default="features", help="name of feature file")
def extract_features(
    dataset, n_workers, mode, dataset_folder, output_folder, output_name
):
    """Extract features from dataset of graphs and save the feature matrix, info and labels"""
    from .io import load_dataset, save_features
    from .feature_extraction import extract

    graphs = load_dataset(dataset, dataset_folder)
    features, features_info = extract(graphs, n_workers=int(n_workers), mode=mode)

    save_features(
        features, features_info, filename=output_name, folder=output_folder,
    )


@cli.command("feature_analysis")
@click.option(
    "-ff", "--feature-folder", default="./results", help="Location of results"
)
@click.option("-fn", "--feature-name", default="features", help="name of feature file")
def feature_analysis(feature_folder, feature_name):
    """Extract features from dataset of graphs"""
    from .io import load_features
    from .feature_analysis import analysis

    features, features_info = load_features(
        filename=feature_name, folder=feature_folder
    )
    analysis(features, features_info)


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
