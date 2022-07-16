from setuptools import setup, find_packages  # noqa: F401

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
    version='0.1.3',
    description='My first Python library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Ivan Dolgushev',
    author_email='ivandolgushev@gmail.com',
    license='MIT',
    packages=["dlid"],
    include_package_data=True,
    install_requires=['numpy', 'torch'],
    tests_require=['pytest'],
    test_suite='tests'
)
