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
    "-n",
    "--n-workers",
    default=1,
    show_default=True,
    help="Number of workers for multiprocessing",
)
@click.option(
    "-m",
    "--mode",
    default="fast",
    show_default=True,
    help="Mode of features to extract (fast, medium, slow)",
)
@click.option(
    "--norm/--no-norm",
    default=False,
    show_default=True,
    help="Normalised features by number of edges/nodes (by default not)",
)
@click.option(
    "-sl",
    "--stats-level",
    default="basic",
    show_default=True,
    help="Level of statistical features (basic, medium, advanced)",
)
@click.option(
    "-of",
    "--output-file",
    help="Location of results, by default same as initial dataset",
)
@click.option(
    "--runtimes/--no-runtimes", default=False, show_default=True, help="Output runtimes"
)
def extract_features(
    dataset, n_workers, mode, output_file, norm, stats_level, runtimes,
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
        output_file = Path(dataset).parent / (Path(dataset).stem + "_features.pkl")

    save_features(
        features, features_info, filename=output_file,
    )


@cli.command("feature_analysis")
@click.argument("feature_file", type=str)
@click.option(
    "-rf",
    "--results-folder",
    default="./results",
    show_default=True,
    help="Location of results",
)
@click.option(
    "--shap/--no-shap",
    default=True,
    show_default=True,
    help="True or False whether to compute shap values",
)
@click.option(
    "-c",
    "--classifier",
    default="RF",
    show_default=True,
    help="classifier feature analysis (RF, LGBM)",
)
@click.option("--kfold/--no-kfold", default=True, show_default=True, help="use K-fold")
@click.option(
    "-p/-np",
    "--plot/--no-plot",
    default=True,
    show_default=True,
    help="Optionnaly plot analysis results",
)
def feature_analysis(feature_file, results_folder, shap, classifier, kfold, plot):
    """Analysis of the features extracted in feature_file"""
    from .io import load_features, save_analysis
    from .feature_analysis import analysis

    [features, features_info] = load_features(filename=feature_file)
    filename_analysis = Path(feature_file).stem + "_analysis"

    X, explainer, shap_values = analysis(
        features,
        features_info,
        filename=filename_analysis,
        folder=results_folder,
        shap=shap,
        classifier_type=classifier,
        kfold=kfold,
        plot=plot,
    )
    save_analysis(
        X, explainer, shap_values, folder=results_folder, filename=filename_analysis
    )


@cli.command("get_data")
@click.argument("dataset_name", type=str)
@click.option(
    "-f",
    "--folder",
    default="./datasets",
    show_default=True,
    help="Location to save dataset",
)
def generate_data(dataset_name, folder):
    """Generate the benchmark or test data

    Dataset_name can be either:
        - TESTDATA: to generate synthetic dataset for testing
        - DD, ENZYMES, REDDIT-MULTI-12K, PROTEINS, MUTAG, 
        or any other dataset hosted on
        https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets

    """
    if dataset_name == "TESTDATA":
        print("--- Building test dataset and creating pickle ---")
        from .dataset_creation import make_test_dataset

        make_test_dataset(folder=folder)
    else:
        print("---Downloading and creating pickle for {}---".format(dataset_name))
        from .dataset_creation import make_benchmark_dataset

        make_benchmark_dataset(dataset_name=dataset_name, folder=folder)
