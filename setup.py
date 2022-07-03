from setuptools import find_packages, setup

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))


# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dlid',
    packages=find_packages(include=['dlid']),
    version='0.1.0',
    description='My first Python library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='ID',
    license='MIT',
    install_requires=['numpy', 'torch'],
    tests_requires=['pytest'],
    test_suite='tests'
)
