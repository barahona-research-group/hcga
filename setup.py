from setuptools import setup, find_packages

setup(
   name='hcga',
   version='0.1',
   description='Highly comparative graph analysis',
   author='Robert Peach + Alexis Arnaudon + Henry Palasciano',
   author_email='r.peach13@imperial.ac.uk',
   packages=['hcga'],  #same as name
   install_requires=['numpy',
                     'scipy',
                     'tqdm', 
                     'networkx',
                     'statsmodels', 
                     'sklearn', 
                     'fa2', 
                     'xgboost', 
                     'seaborn'], #external packages as dependencies
   include_package_data = True
)
