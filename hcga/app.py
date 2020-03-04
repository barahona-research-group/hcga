""" hcga app with click module """
import click
from pathlib import Path

from .io import load_graphs, save_features, load_features
from .feature_extraction import extract
from .feature_analysis import analysis
import numpy

numpy.seterr(all="ignore")
import warnings

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
@click.option("-f", "--folder", default="./datasets", help="Location of dataset")
def extract_features(dataset, n_workers, mode):
    """Extract features from dataset of graphs"""
    graphs, labels = load_datasets(dataset)
    feature_matrix, features_info = extract(graphs, n_workers=int(n_workers), mode=mode)
    data_file = Path(dataset)
    feature_filename = data_file.parent / (data_file.stem + "_features.pkl")
    save_features(feature_matrix, features_info, labels, filename=feature_filename)


@cli.command("feature_analysis")
@click.argument("feature_file", type=str)
def feature_analysis(feature_file):
    """Extract features from dataset of graphs"""
    feature_matrix, features_info, labels = load_features(feature_file)
    analysis(feature_matrix, features_info, labels)


@cli.command("get_benchmark_data")
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
