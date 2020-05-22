"""Setup file for PyRCCA"""

from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / 'README.md'
with open(readme_path, encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyrcca',
    version='0.1',
    author='Bilenko et al',
    packages=find_packages(),
    url = 'https://github.com/gallantlab/pyrcca',
    license='Free for non-commercial use',
    description='Regularized kernel canonical correlation analysis in Python.',
    long_description_content_type='text/markdown',
    long_description=long_description,
    install_requires=[
	'h5py',
	'numpy',
	'scipy',
    	]
    )
