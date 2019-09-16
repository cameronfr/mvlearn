dataDir = "/mnt/disks/mindbrainbodydata/"
#MNI2mm from https://github.com/Jfortin1/MNITemplate/raw/master/inst/extdata/MNI152_T1_2mm.nii.gz
# http://sci-hub.tw/https://www.sciencedirect.com/science/article/pii/S0010945219300012 -- talks about advantages
# of ICA for finding resting state networks, but also mentions finding DMN using seed method w/ MNI coords (-1, 47, 4) for mPFC seeed

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as plg
import nibabel as nib
import skvideo.io
import visdom
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import torch
import tqdm
import sklearn.decomposition
import scipy.ndimage
import skimage.transform
import matplotlib.animation as manimation
from matplotlib.animation import FuncAnimation
import scipy.misc
from IPython.display import Video
# vis = visdom.Visdom()

from plotly import offline as py
import plotly.tools as tls
py.init_notebook_mode()
plotly = lambda: py.iplot(tls.mpl_to_plotly(plt.gcf())) #usage: put plotly() under plt.hist(...) and run the two lines together

%matplotlib inline
plt.style.use("ggplot")
%config InlineBackend.figure_format = 'svg'

#------------------------------ Loading the reference

# MNI1mm shape is (181, 217, 181). idx0: X: left to right. idx1: posterior to anterior idx2: inferior to superior
# Origin in MNI1mm is (+90, +126, +72)
# hard to google MNI info -- e.g. fact that MNI could mean 1mm or 2mm, where the origin is (had to use neurosynth to find)
# the references are in MNI2mm so have a shape that is half the size of MNI1mm
def MNICoordToArrayCoords(pos):
    originAdded = pos + np.array([90, 126, 72])
    toTwoMM = (originAdded / 2.0).astype(np.int)
    return toTwoMM

mniReference = nib.load("/home/cameronfranz/MindBrainBodyData/MNI152_T1_2mm_Brain.nii.gz")
mniReference = mniReference.get_fdata()
coords = np.where(mniReference > 0)
colors = mniReference[coords]
color3dScatterPlot(coords, colors)
#MNI reference has points not on brain equal to zero
fmriSliceToVideo(mniReference[:, : ,:] == 0, 10)
fmriSliceToVideo(mniReference[:, : ,:], 10)
#verification of mask usability shows that there are a couple holes in the mask -- looks like ventricles, though


#------------------------------ Spatial Smoothing (MindBrainBody preprocessing does not do this)

fmriSubj2 = nib.load("/home/cameronfranz/MindBrainBodyData/rawData/sub-010002/ses-01/func/sub-010002_ses-01_task-rest_acq-AP_run-01_bold.nii.gz")
fmriSubj2Dat  = fmriSubj2.get_fdata()

plt.plot(fmriSub2Dat[40, 40, 32])
plt.plot(movingAverage(fmriSub2Dat[40, 40, 32], 20))

t = np.linspace(-10, 10, 10)
gauss1d = np.exp(-0.1*t**2)
gauss1d /= np.trapz(gauss1d) # what's diff between trapz and sum?
gauss2d = (gauss1d[:, np.newaxis] * gauss1d[np.newaxis, :])
gauss3d = gauss2d[:, :, np.newaxis] * gauss2d[np.newaxis, :, :] * gauss2d[:, np.newaxis, :]
# plt.imshow(gauss3d[:, 0, :], vmin=0, vmax=0.000001) #if using imshow for 3d, use consistent vmin/vmax or will get confused
# color3dScatterPlot(np.where(gauss3d > 1e-10), gauss3d[np.where(gauss3d > 1e-10)])

weight = gauss3d[np.newaxis, np.newaxis, :, :, :]
inputTensor = fmriSubj2Dat.transpose((3, 0, 1, 2,))[:, np.newaxis, :, :, :] # put time dim as batch dim and set channel dim to 1
inputTensor.shape

convolved = torch.nn.functional.conv3d(torch.FloatTensor(inputTensor[:10]), torch.FloatTensor(weight))
convolved = scipy.ndimage.filters.convolve(inputTensor, weight)

# torch is kinda slow and complains about running about of memory. scipy.ndimage.filters.convolve might work better
# also, not sure what kernel sizes / stdev are normally used.

#------------------------------ Code that loads each subject

allCorrelations = []
correlationThresholds = []

for i in range(10):

    os.chdir(dataDir)
    subjectFolders = next(os.walk('.'))[1]

    # subjectID = subjectFolders[1]
    subjectID = subjectFolders[i]

    # NOTE: raw != native -- the raw data is a different download.
    # nativeFmriObj = nib.load(os.path.join(dataDir, subjectID, "func/" + subjectID + "_ses-01_task-rest_acq-AP_run-01_native.nii.gz"))
    # nativeFmriDat = nativeFmriObj.get_fdata()
    processedFmriObj = nib.load(os.path.join(dataDir, subjectID, "func/" + subjectID + "_ses-01_task-rest_acq-AP_run-01_MNI2mm.nii.gz"))
    processedFmriDat = processedFmriObj.get_fdata()
    # plt.plot(processedFmriDat[30, 30, 30, :].transpose())


    #------------------------------ mPFC seed for visualizing DMN
    mPFCLocation = MNICoordToArrayCoords([-1, 47, 4])
    # plt.grid(False)
    # plt.imshow(mniReference[:, :, mPFCLocation[2]]) #imshow has first dim in vertical dir
    # plt.annotate('mPFC seed', xy=(mPFCLocation[1], mPFCLocation[0]), xycoords='data',
    #              xytext=(0.5, 0.5), textcoords='figure fraction',
    #              arrowprops=dict(facecolor='black', width=3, headwidth=7))

    # get null distribution of correlations between voxels by randomly sampling

    # From exploring, can see that max vs min of the BOLD signal looks normally distributed with mean 5 and stdev ~0.5
    # note that this is the processed data, so could be result of preprocessing
    # found https://ftp.nmr.mgh.harvard.edu/pub/docs/SavoyfMRI2014/fmri.april2011.pdf on preprocessing
    # Even better, the paper lists their preprocessing steps https://www.nature.com/articles/sdata2018308 in section titled preprocessing and have pipeline on github
    # Paper mentions mean-centering and variance-normalization but I only see mean-centering -- will have to look at pipeline to see what the variance-normalization is

    def randomCoords(shape):
        c = []
        for dimSize in shape:
            c.append(np.random.randint(dimSize))
        return c
    def randomCoordsWithinMNIMask():
        coords = randomCoords(mniReference.shape)
        while mniReference[tuple(coords)] == 0:
            coords = randomCoords(mniReference.shape)
        return coords

    # visualize distribution of properties of the time series across voxels
    # for i in range(10):
    #     coords = randomCoordsWithinMNIMask()
    #     plt.plot(processedFmriDat[tuple(coords)])
    # ranges = []
    # variances = []
    # means = []
    # for i in range(1000):
    #     coords = randomCoordsWithinMNIMask()
    #     timeSeries = processedFmriDat[tuple(coords)]
    #     ranges.append(np.max(timeSeries) - np.min(timeSeries))
    #     variances.append(np.var(timeSeries))
    #     means.append(np.mean(timeSeries))
    # plt.hist(ranges, bins=20)
    # plt.hist(variances, bins=20)
    # plt.hist(means, bins=20)

    # was normalizing the variance to 1 across voxels, not sure if it makes a difference since already pretty normalized
    # processedAgainFmriDat = processedFmriDat / (np.std(processedFmriDat, axis=3)[:, :, :, np.newaxis])
    processedAgainFmriDat = processedFmriDat

    # create a basic null distribution. Will need to look into "FWE cluster correction", other correlation metrics, better null distribution ideas.
    pearsonCorrelationNullDist = []
    for i in range(50000):
        coords1 = randomCoordsWithinMNIMask()
        coords2 = randomCoordsWithinMNIMask()
        timeSeries1 = processedFmriDat[tuple(coords1)]
        timeSeries2 = processedFmriDat[tuple(coords2)]
        correlation = np.mean(timeSeries1 * timeSeries2)
        pearsonCorrelationNullDist.append(correlation)
    pearsonCorrelationNullDist = np.sort(pearsonCorrelationNullDist)
    # plt.hist(pearsonCorrelationNullDist, bins=20)
    pValue = lambda v: 1 - np.searchsorted(pearsonCorrelationNullDist, v, side="left",) / pearsonCorrelationNullDist.shape[0]
    correlationAtGivenPValue = lambda p: pearsonCorrelationNullDist[-int(p * pearsonCorrelationNullDist.shape[0])]
    correlationAtGivenPValue(0.01)

    # calculate correlation of seed location with every other location
    # plt.plot(processedFmriDat[tuple(mPFCLocation)])
    # fmriSliceToVideo(processedFmriDat[:, :, mPFCLocation[2], :], 10)
    dotProd = (processedAgainFmriDat * processedAgainFmriDat[tuple(mPFCLocation)])
    correlationsToSeed = np.mean(dotProd, axis=3)
    significantCorrelationToSeedMask = correlationsToSeed > correlationAtGivenPValue(0.001)
    # np.sum(significantCorrelationToSeedMask)
    # fmriSliceToVideo(significantCorrelationToSeedMask, 10)
    # color3dScatterPlot(np.where(significantCorrelationToSeedMask), significantCorrelationToSeedMask[np.where(significantCorrelationToSeedMask)])
    allCorrelations.append(correlationsToSeed)
    correlationThresholds.append(correlationAtGivenPValue(0.001))

allCorrelations[2].shape
correlationThresholds

totalMask = allCorrelations[0] > correlationThresholds[0]
for i in range(1, 10):
    array = allCorrelations[i]
    signifMask = array > correlationThresholds[i]
    totalMask = totalMask | signifMask
    # color3dScatterPlot(np.where(signifMask), signifMask[np.where(signifMask)])

totalMask.shape
coords = np.where(mniReference > 0)
colors = totalMask[coords]
np.sum(colors.astype(np.int))
# color3dScatterPlot(np.where(totalMask), totalMask[np.where(totalMask)])
color3dScatterPlot(coords, colors.astype(np.int)) # no good
coloredMNI = mniReference.copy()
coloredMNI[totalMask] = 10000
fmriSliceToVideo(coloredMNI, 10)
plt.grid(False)
plt.imshow(coloredMNI[:, :, MNICoordToArrayCoords([0, 0, 40])[2]])
plt.imshow(np.flip(np.flip(coloredMNI[MNICoordToArrayCoords([0, 0, 40])[0], :, :].transpose(), axis=0), axis=1))



#------------------------------ SVDD Attempts
fmriSliceToVideo(processedFmriDat[:, :, 30, :], 10)
processedFmriDat.reshape(-1, 652).shape
processedFmriDat.shape
import scipy.sparse
sdvdd = scipy.sparse.linalg.svds(processedFmriDat.reshape(-1, 652), k=5)
sdvdd[0].shape
niceheatmap = sdvdd[0].reshape(*processedFmriDat.shape[:3], 5)

niceheatmap = niceheatmap[:, :, :, 0]
plt.hist(niceheatmap.ravel())
coords = np.where(niceheatmap > 0.1

plt.hist(np.exp(niceheatmap).ravel())
fmriSliceToVideo(np.exp(niceheatmap) - 1, 10)
fmriSliceToVideo(mniReference, 10)


#------------------------------ Utility functions

def color3dScatterPlot(coords, colors):
    fig = plg.Figure(data=[plg.Scatter3d(x=coords[0], y=coords[1], z=coords[2],
                                    mode='markers',
                                    hovertext=colors,
                                    hoverinfo="text",
                                       marker=dict(
                                        size=3.5,
                                        color=colors,#values[:, 0],
                                        colorscale='Viridis',
                                        opacity=0.6
                                       )
                                       )])
    fig.update_layout(scene_aspectmode='data')
    fig.update_layout(autosize=True, width=800, height=600)
    return fig

def fmriSliceToVideo(threeDimSlice, fps):
    cmap = matplotlib.cm.get_cmap("viridis")
    threeDimSlice = np.copy(threeDimSlice)
    # threeDimSlice = np.exp(threeDimSlice/500) #make diff in higher brigh lvl show more signif than diff in lower bright lvl. Not what we want.
    threeDimSlice = (threeDimSlice / np.max(threeDimSlice))
    threeDimSlice = cmap(threeDimSlice)[:, :, :, :3]
    threeDimSlice = (threeDimSlice * 255).astype(np.uint8)
    threeDimSlice = threeDimSlice.transpose(2, 0, 1, 3)

    # # slower, has the graph. 31s for 100 frames.
    # fig, ax = plt.subplots(1)
    # ax.grid(False)
    # img = ax.imshow(threeDimSlice[0, :, :, 0])
    # def updateImg(i):
    #     img.set_array(threeDimSlice[i, :, :, 0])
    #     return img, "test"
    # animation = FuncAnimation(fig, updateImg, frames=range(20), interval=5, blit=False)
    # animation.save("outputvideotmp.mp4", fps = 2, dpi=600)

    writer = skvideo.io.FFmpegWriter("outputvideotmp.mp4", outputdict={'-r': str(fps), '-vcodec': 'libx264'})
    for i in range(threeDimSlice.shape[0]):
        slice = threeDimSlice[i, :, :, :]
        slice.shape
        slice = scipy.misc.imresize(slice, 4.0, interp="nearest")
        writer.writeFrame(slice)
    writer.close()
    return Video("./outputvideotmp.mp4", embed=True, )
