pyrcca
======

Regularized kernel canonical correlation analysis in Python.

In this startup example, we create two random datasets with two latent variables, and use Pyrcca to implement CCA between them. The datasets are broken up into two halves. First, we use the first half of the datasets to train a CCA mapping. Then, we test the found mapping we found by validating it on the second half of the datasets. This procedure assures that the found canonical variates are generalizable and are not overfitting to the training data.

```python
import numpy as np
import rcca

nObservations = 1000

# Define two latent variables
lat_var1 = np.random.randn(nObservations,)
lat_var2 = np.random.randn(nObservations,)

# Define independent signal components
indep1 = np.random.randn(nObservations, 4)
indep2 = np.random.randn(nObservations, 4)

# Define two datasets as a combination of latent variables
# and independent signal components
data1 = indep1 + np.vstack((lat_var1, lat_var1, lat_var2, lat_var1)).T
data2 = indep2 + np.vstack((lat_var1, lat_var1, lat_var2, lat_var1)).T

# Divide data into two halves: training and testing sets
train1 = data1[:nObservations/2]
test1 = data1[nObservations/2:]
train2 = data2[:nObservations/2]
test2 = data2[nObservations/2:]

# Set up Pyrcca
cca = rcca.CCA(kernelcca=False, numCC=2, reg=0.)

# Find canonical components
cca.train([train1, train2])

# Test on held-out data
corrs = cca.validate([test1, test2])
```