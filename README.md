pyrcca
======

Regularized kernel canonical correlation analysis in Python

```
$ python
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