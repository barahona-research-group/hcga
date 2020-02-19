""" hcga app with click module """
import click

from .io import load_graphs, save_features
from .feature_extraction import extract

@click.group()
def cli():
    """init app"""

@cli.command("extract_features")
@click.argument("dataset", type=str)
@click.option("-n", "--n-workers", help="Number of workers")
def extract_features(dataset, n_workers=1):
    """extract features from dataset of graphs"""
    graphs = load_graphs(dataset) 
    feature_matrix, features_info = extract(graphs, n_workers=int(n_workers))
    save_features(feature_matrix, features_info)



