""" hcga app with click module """
import click

from .io import load_graphs, save_features, load_features
from .feature_extraction import extract
from .feature_analysis import analysis


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
@click.option("-n", "--n-workers", help="Number of workers")
def feature_analysis(filename='features.pkl', n_workers=1):
    """extract features from dataset of graphs"""
    feature_matrix, features_info, labels= load_features(filename)
    shap_values = analysis(feature_matrix, features_info, labels)
    


