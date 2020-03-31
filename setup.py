from setuptools import setup, find_packages

setup(
   name='hcga',
   version='1.0.0',
   description='Highly comparative graph analysis',
   author='Robert Peach + Alexis Arnaudon + Henry Palasciano',
   author_email='r.peach13@imperial.ac.uk',
   packages=find_packages(),
   install_requires=[
                    'click',
                    'numpy',
                     'scipy',
                     'tqdm', 
                     'networkx',
                     'statsmodels', 
                     'sklearn', 
                     'fa2', 
                     'xgboost', 
                     'seaborn', 
                     'lightgbm',
                     'shap',
                     'pandas',
                     'wget'],
    entry_points={
        'console_scripts': ['hcga=hcga.app:cli'],
    }
)
