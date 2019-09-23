# Using https://yugeten.github.io/posts/2019/09/GP/ as a guide for exploration (explanations and equations, no code)
# good to think of a gaussian process as a definer of a space of functions
#TODO: understand derivation of marginal http://fourier.eng.hmc.edu/e161/lectures/gaussianprocess/node7.html

import numpy as np
import matplotlib.pyplot as plt

from plotly import offline as py
import plotly.tools as tls
py.init_notebook_mode()
plotly = lambda: py.iplot(tls.mpl_to_plotly(plt.gcf())) #usage: put plotly() under plt.hist(...) and run the two lines together

%matplotlib inline
plt.style.use("ggplot")
%config InlineBackend.figure_format = 'svg'

#------------------------------ Basic Gaussian Processes Implementation ------------------------------

# RBF kernel and hyperparameters.
sigma = 100
l = 10
kernelFunction = lambda x: sigma**2 * np.exp((-1.0/(2*(l**2))) * (x[:, :, 0]-x[:, :, 1])**2)

# Current only working with one-dim case. n-dim afaik only diff is (n-dim -> R) kernel function => O(n^dim) (in addition to the inverse, which is O(n^3))
def samplePointsFromGPGivenKnownPoints(knownPointXs, knownPointYs, unknownPointXs, numSamples):
    # Below sets the covariances based on the kernel and the x-values of the points
    knownUnknownXsPairs = np.stack(np.broadcast_arrays(knownPointXs[:, np.newaxis], unknownPointXs[np.newaxis, :]), axis=2)
    knownUnknownPointsCovar = kernelFunction(knownUnknownXsPairs)
    knownXsPairs = np.stack(np.broadcast_arrays(knownPointXs[:, np.newaxis], knownPointXs[np.newaxis, :]), axis=2)
    knownPointsCovar = kernelFunction(knownXsPairs)
    unknownXsPairs = np.stack(np.broadcast_arrays(unknownPointXs[:, np.newaxis], unknownPointXs[np.newaxis, :]), axis=2)
    unknownPointsCovar = kernelFunction(unknownXsPairs)

    # Below calculates p(unknown points | known points), where both are gaussian dist vectors, using block-marginalization technique thing
    knownPointsCovarInverse = np.linalg.inv(knownPointsCovar)
    marginalMean = 0 + knownUnknownPointsCovar.transpose() @ knownPointsCovarInverse @ (knownPointYs - 0)
    marginalCovar = unknownPointsCovar - knownUnknownPointsCovar.transpose() @ knownPointsCovarInverse @ knownUnknownPointsCovar
    sampledPoints = np.random.multivariate_normal(marginalMean, marginalCovar, numSamples)
    return sampledPoints

# Testing on example with RBF kernel.
knownPointXs = np.array([1, 3, 10, 15, 18])
knownPointYs = np.array([5, 6, 3, 12, 8])
unknownPointXs = np.linspace(0, 20, 100)
sampledPoints = samplePointsFromGPGivenKnownPoints(knownPointXs, knownPointYs, unknownPointXs, 1000)

plt.plot(unknownPointXs, sampledPoints[:10, :].transpose(), color=(0.3, 0.3, 1, 0.5))
plt.plot(knownPointXs, knownPointYs, marker="o", alpha=0.8)
plt.show()

# can't use below for testing, because sampleCovs are the cov of the p(unknown | known) dist, not the individual cov of p(unknown) dist
expectedCovariances = kernelFunction(np.stack(np.broadcast_arrays(unknownPointXs[:, np.newaxis], unknownPointXs[np.newaxis, :]), axis=2))
sampleCovs = np.cov(sampledPoints.transpose()) # covar arr 1st dim is variable, second observation
sampleMeans = np.mean(sampledPoints, axis=0)

#------------------------------ Messing around below ------------------------------

covar = np.eye(10) * 1
covar[0,1] = covar[1,0] = 0.9
dist = np.random.multivariate_normal(np.zeros(10), covar, size=100000)
linCombOfComponents = np.sum((np.random.random(10).reshape(1,-1) * dist), axis=1)
# plt.hist(linCombOfComponents, bins=30) #by definition of multivariate normal, this linComb should be normal.

# 2d histogram = heatmap
heatmap = np.zeros((100, 100))
heatcenter = np.round(np.array(heatmap.shape)/2).astype(np.int)
coords = np.floor(dist[:, :2] * 10).astype(np.int)
np.max(coords)
np.add.at(heatmap, (coords[:, 0]+heatcenter[0], coords[:, 1]+heatcenter[0],), 1) # if index presented twice, add.at will increment it twice
plt.grid(False)
plt.imshow(heatmap)
# plt.hist2d(dist[:, 0], dist[:, 1], bins=60) # equivalent to doing above heatmap stuff
plt.show()

# covariance
covar = 0.9
np.linalg.inv(np.array([[1,covar],[covar,1]]))
x = [10,0]
x = [10, 10]
c = 1000000
exponentVal = c*x[0]**2 - 2*c*x[0]*x[1] + c*x[1]**2
exponentVal
# as c-> inf, the above equation is 0 if x[1] = x[2], a large # otherwise. not obv to me.

# plotting in a different way
covar = np.eye(10)
covar[0,1] = covar[1,0] = 0.9
dist = np.random.multivariate_normal(np.zeros(10), covar, size=100000)
# two points, compare the two dims with covar to two dims without covar
plt.ylim(-3, 3)
# plt.plot(dist[np.random.randint(dist.shape[0]),:2])
plt.plot(dist[np.random.randint(dist.shape[0]),1:3])
#ten points
plt.ylim(-3, 3)
plt.plot(dist[np.random.randint(dist.shape[0]),:])

# looking at RBF that defines covariance
sigma = 1
l = 10
kernelFunction = lambda x1, x2: sigma**2 * np.exp((-1.0/(2*(l**2))) * (x1-x2)**2)
covarianceMatrix = np.fromfunction(kernelFunction, shape=(100, 100))
plt.imshow(covarianceMatrix)

# below is what the function signature for a gaussian process should look like.
# gaussianProcess(mean = lambda x: ..., covar = lambda x1, x2: ...) -> PRIOR of distribution of functions

# sampling functions randomly from this prior
sigma = 1
l = 10
kernelFunction = lambda x1, x2: sigma**2 * np.exp((-1.0/(2*(l**2))) * (x1-x2)**2)
points = np.zeros(50)
# case: we have only points[0]
points[0] = 10
points[1] = #problem: what does the distribution of this point look like p(point2 | point1)
# case: we have known points at idxs [...] and want to sample from unknown points.
