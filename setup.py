from setuptools import setup

setup(
   name='hcga',
   version='1.0',
   description='Highly comparative graph analysis',
   author='Robert Peach + Alexis Arnaudon',
   author_email='r.peach13@imperial.ac.uk',
   packages=['hcga'],  #same as name
   install_requires=['networkx','numpy'], #external packages as dependencies
)
