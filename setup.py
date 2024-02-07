from setuptools import setup, find_packages

reqs = [
    "click>=7.1.1",
    "numpy>=1.18.2",
    "scipy>=1.11",
    "tqdm>=4.45.0",
    "networkx>=3.0",
    "scikit-learn>=0.23.1",
    "matplotlib>=1.4.3",
    "seaborn>=0.9.0",
    "shap>=0.35.0",
    "pandas>=1.0.3",
    "wget>=3.2",
    "sympy>=1.4",
    "joblib>=0.14.1",
    "IPython>=7.19.0",
    "xgboost>=1.3.3",
]
doc_reqs = [
    "sphinx",
    "sphinx_click",
]

test_reqs = [
    "pytest>=7",
]

setup(
    name="hcga",
    version="1.0.2",
    description="Highly comparative graph analysis",
    author="Robert Peach + Alexis Arnaudon + Henry Palasciano",
    author_email="r.peach13@imperial.ac.uk",
    packages=find_packages(),
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
    },
    entry_points={"console_scripts": ["hcga=hcga.app:cli"]},
)
