""" hcga app with click module """
import click

from .io import load_graphs, save_features, load_features
from .feature_extraction import extract
from .feature_analysis import analysis
from .make_data import make_benchmark_dataset, make_test_data


@click.group()
def cli():
    """init app"""


@cli.command("extract_features")
@click.argument("dataset", type=str)
@click.option("-n", "--n-workers", help="Number of workers")
def extract_features(dataset, n_workers=1):
    """extract features from dataset of graphs"""
    graphs, labels = load_graphs(dataset)
    feature_matrix, features_info = extract(graphs, n_workers=int(n_workers))
    save_features(feature_matrix, features_info, labels)


@cli.command("feature_analysis")
def feature_analysis(filename="features.pkl"):
    """extract features from dataset of graphs"""
    feature_matrix, features_info, labels = load_features(filename)
    analysis(feature_matrix, features_info, labels)

@cli.command('get_benchmark_data')
@click.argument("dataset_name", type=str) 
@click.option("-o", "--output", default='./', help='Location to save the data')
def generate_data(dataset_name, output):
    """Generate the benchmark or test data"""
    output = output + '/'
    if dataset_name =='TESTDATA':
        print("--- Building test dataset and creating pickle ---")
        make_test_data(directory=output, save_data=True)
    else:
        print("---Downloading and creating pickle for {}---".format(dataset_name))
        make_benchmark_dataset(data_name=dataset_name, director=output)

