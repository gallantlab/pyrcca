"""Setup file for PyRCCA"""

from setuptools import setup, find_packages
from pathlib import Path
import re

readme_path = Path(__file__).parent / 'README.md'
with open(readme_path, encoding='utf-8') as f:
    long_description = f.read()

# get version from rcca/__init__.py
__version__ = 0.0
with open('rcca/__init__.py') as f:
    infos = f.readlines()
for line in infos:
    if "__version__" in line:
        match = re.search(r"__version__ = '([^']*)'", line)
        __version__ = match.groups()[0]

setup(
    name='pyrcca',
    version=__version__,
    author='Bilenko et al',
    packages=find_packages(),
    url='https://github.com/gallantlab/pyrcca',
    license='Free for non-commercial use',
    description='Regularized kernel canonical correlation analysis in Python.',
    long_description_content_type='text/markdown',
    long_description=long_description,
    install_requires=[
        'h5py',
        'numpy',
        'scipy',
        'joblib',
    ],
)
