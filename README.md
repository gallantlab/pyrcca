pyrcca
======

Regularized kernel canonical correlation analysis in Python.

A static Jupyter notebook with the analysis of the example below can be found <a href="https://github.com/gallantlab/pyrcca/blob/master/Pyrcca_usage_example.ipynb">here</a>.

A static Jupyter notebook with Pyrcca analysis of fMRI data can be found <a href="https://github.com/gallantlab/pyrcca/blob/master/Pyrcca_neuroimaging_example.ipynb">here</a>.

Both notebooks can be explored interactively by cloning this repository.


For more information, consult the following e-print publication:
Bilenko, N.Y. and Gallant, J.L. (2015). Pyrcca: regularized kernel canonical correlation analysis in Python and its applications to neuroimaging. Frontiers in Neuroinformatics <a href="http://journal.frontiersin.org/article/10.3389/fninf.2016.00049/abstract"> doi: 10.3389/fninf.2016.00049</a>


In this startup example, two artificially constructed datasets are created. The datasets depend on two latent variables. Pyrcca is used to find linear relationships between the datasets. 

```python
# Imports
import numpy as np
import rcca

# Initialize number of samples
nSamples = 1000

# Define two latent variables (number of samples x 1)
latvar1 = np.random.randn(nSamples,)
latvar2 = np.random.randn(nSamples,)

# Define independent components for each dataset (number of observations x dataset dimensions)
indep1 = np.random.randn(nSamples, 4)
indep2 = np.random.randn(nSamples, 5)

# Create two datasets, with each dimension composed as a sum of 75% one of the latent variables and 25% independent component
data1 = 0.25*indep1 + 0.75*np.vstack((latvar1, latvar2, latvar1, latvar2)).T
data2 = 0.25*indep2 + 0.75*np.vstack((latvar1, latvar2, latvar1, latvar2, latvar1)).T

# Split each dataset into two halves: training set and test set
train1 = data1[:nSamples/2]
train2 = data2[:nSamples/2]
test1 = data1[nSamples/2:]
test2 = data2[nSamples/2:]

# Create a cca object as an instantiation of the CCA object class. 
cca = rcca.CCA(kernelcca = False, reg = 0., numCC = 2)

# Use the train() method to find a CCA mapping between the two training sets.
cca.train([train1, train2])

# Use the validate() method to test how well the CCA mapping generalizes to the test data.
# For each dimension in the test data, correlations between predicted and actual data are computed.
testcorrs = cca.validate([test1, test2])
```
