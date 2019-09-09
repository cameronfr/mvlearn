%cd ~/MindBrainBodyData/ds000221_R1.0.0

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
import skimage.transform
import matplotlib.animation as manimation
from matplotlib.animation import FuncAnimation
import scipy.misc
from IPython.display import Video
vis = visdom.Visdom()

from plotly import offline as py
import plotly.tools as tls
py.init_notebook_mode()
plotly = lambda: py.iplot(tls.mpl_to_plotly(plt.gcf())) #usage: put plotly() under plt.hist(...) and run the two lines together

%matplotlib inline
plt.style.use("ggplot")
%config InlineBackend.figure_format = 'svg'

participantsMetadata = pd.read_csv("participants.tsv", sep="\t")
ages = participantsMetadata.iloc[:, 2].tolist()
ages = np.array(ages)
np.sum(ages == "n/a")
ages = ages[ages != "n/a"]
ages = [int(text.split("-")[0]) + 2.5 for text in ages]
plt.hist(ages, bins=np.linspace(0, 100, 20))
plotly()
gender = participantsMetadata.iloc[:, 1].tolist()
gender = np.array(gender)
np.sum(gender == "M")/gender.shape[0]

# subject 1 has no ses1
fmriSub2 = nib.load("sub-010002/ses-01/func/sub-010002_ses-01_task-rest_acq-AP_run-01_bold.nii.gz")
fmriSub2.shape
fmriSub2.affine.shape
fmriSub2.header.get_xyzt_units()
fmriSub2.header.get_slice_times()
fmriSub2Dat = fmriSub2.get_fdata()

fmriSub2Preprocessed = nib.load("../preprocessedSubj1Data/func/sub-032301_ses-01_task-rest_acq-AP_run-01_MNI2mm.nii.gz")
fmriSub2Preprocessed = fmriSub2Preprocessed.get_fdata()
fmriSub2Preprocessed.shape

for i in range(20):
    slice = fmriSub2Dat[:, :, 32, i]
    slice = np.exp(slice/500)
    plt.grid(False)
    plt.imshow(slice)
    plt.show()
    time.sleep(0.1)

threeDimSlice = fmriSub2Dat[:, :, 30, :]
threeDimSlice = fmriSub2Dat[:, :, :, 30].transpose((0, 2, 1))
fmriSliceToVideo(threeDimSlice, 10)

plt.plot(fmriSub2Dat[30, 30, 30, :])
plt.plot(fmriSub2Dat[20, 20, 20, :])
plotly()

#find the linear combinations of areas that have the maximum variance over time.
pcaDecomp = sklearn.decomposition.PCA(n_components=100).fit_transform(fmriSub2Dat.reshape((-1, fmriSub2Dat.shape[-1])).transpose())
for i in range(10):
    plt.plot(movingAverage(pcaDecomp[:, i], 20))

for i in range(100):
    randVoxelTimeSeries = fmriSub2Dat[np.random.randint(88), np.random.randint(88), np.random.randint(64), :]
    plt.plot(movingAverage(randVoxelTimeSeries, 5))

def movingAverage(oneDimArr, windowSize):
    oneDimArr = torch.FloatTensor(oneDimArr).view(1, 1, -1)
    kernel = (torch.ones(windowSize) / windowSize).view(1, 1, -1)
    averaged = torch.nn.functional.conv1d(oneDimArr, kernel).squeeze().numpy()
    return averaged

#------------------------------ autocorrelatnion ------------------------------
fmriSub2Dat.shape
fmriSub2Dat2 = fmriSub2Dat[:, :, :, :10]

# want: (88*88*64) by (88*88*64) matrix of correlations. would be like 1TB -- possible but not practical.
# autoCorr = np.matmul(fmriSub2Dat[:, :, :, :, np.newaxis].transpose((0, 1, 2, 4, 3)), fmriSub2Dat[:, :, :, :, np.newaxis], )

np.sum(np.power(fmriSub2Dat[30, 30, 0], 2))


# Try and view basic 3d structure with scatter3d
# slice = fmriSub2Dat[:, :, 30, 0]
# plt.imshow(slice < 200)
coords = np.where(fmriSub2Dat[:, :, :30, 0] > 200)
values = fmriSub2Dat[coords[0], coords[1], coords[2], :]
totalVarOverTimePerVoxel = np.var(values, axis=1)
totalVarOverTimePerVoxel = np.clip(totalVarOverTimePerVoxel, 0, np.percentile(totalVarOverTimePerVoxel, 95)) #spread still not good for visuals
totalVarOverTimePerVoxel = np.log(totalVarOverTimePerVoxel)
plt.hist(totalVarOverTimePerVoxel)
fmriVarData = np.zeros_like(fmriSub2Dat[:, :, :, 0])
fmriVarData[coords[0], coords[1], coords[2]] = totalVarOverTimePerVoxel
color3dScatterPlot(coords, totalVarOverTimePerVoxel)
fmriSliceToVideo(fmriVarData, 10)

# Cool anat Nii in dataset
anatSub2 = nib.load("sub-010002/ses-01/anat/sub-010002_ses-01_inv-1_mp2rage.nii.gz")
anatSub2 = nib.load("sub-010002/ses-01/dwi/sub-010002_ses-01_dwi.nii.gz")
anatSub2 = anatSub2.get_fdata()
plt.imshow(anatSub2[:, :, 30, 0])
fmriSliceToVideo(anatSub2[:, :, :, 0], 10)
coords = np.where(anatSub2[::2, ::2, ::2, 0] > 400)
values = anatSub2[::2, ::2, ::2, 0][coords[0], coords[1], coords[2]]
plt.hist(values); plotly()
color3dScatterPlot(coords, values)

fmriSliceToVideo(anatSub2[:, :, :, 0], 10)

totalVarOverTimePerVoxel = np.var(fmriSub2Dat, axis=3)
totalVarOverTimePerVoxel = np.clip(totalVarOverTimePerVoxel, 10, np.percentile(totalVarOverTimePerVoxel, 95)) #spread still not good for visuals
totalVarOverTimePerVoxel = np.log(totalVarOverTimePerVoxel)
plt.hist(totalVarOverTimePerVoxel.ravel())
threeDimSlice = totalVarOverTimePerVoxel[:, :, :]
fmriSliceToVideo(threeDimSlice, 10)



# neurodata advice: propose SOMETHING
# registering to atlas
# trying to get visuals of activity over time
# load multiple of these fmri samples
# try and get picture of default mode network? PCA for something?
# stats of activity based on fmri data segregated into atlas regions (e.g. hippocampus)
#------------------------------ BULK PROCESSING ------------------------------
subFolders = list(filter(lambda x: "sub-" in x, os.listdir("./"))) #9 subjects
subFMRIArrays = []
for folder in tqdm.tqdm(subFolders):
    fmriPath1 = os.path.join(folder, "ses-02/func/" + folder + "_ses-02_task-rest_acq-AP_run-01_bold.nii.gz")
    fmriPath2 = os.path.join(folder, "ses-01/func/" + folder + "_ses-01_task-rest_acq-AP_run-01_bold.nii.gz")
    fmriPath = fmriPath1 if os.path.isfile(fmriPath1) else fmriPath2
    fmriData = nib.load(fmriPath).get_fdata()
    subFMRIArrays.append(fmriData)

for arr in subFMRIArrays:
    varArr = np.var(arr, axis=3)
    varArr = np.clip(varArr, 0, np.percentile(varArr, 95)) #spread still not good for visuals
    # varArr = np.log(varArr)
    plt.grid(False)
    # plt.imshow(arr[:, :, 30, 0])
    plt.imshow(varArr[:, :, 30])
    plt.show()





# Check out all the different fmap ant anat Nii Files
folderPath = "./sub-010002/ses-01/fmap/"
folderPath = "./sub-010002/ses-01/anat/"
niiFileTypes = filter(lambda x: "nii" in x, os.listdir(folderPath))
niiFileTypes = list(niiFileTypes)
nii = niiFileTypes[6]
niiFileTypes
path = os.path.join(folderPath, nii)
loadedNii = nib.load(path)
loadedNii = loadedNii.get_fdata()
loadedNii.shape
plt.grid(False)
plt.imshow(loadedNii[:, :, 32, 0])
fmriSliceToVideo(loadedNii, 10)

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

# interface with four sliders would probably be better...
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

def plotlyAnimationAttempt():
    #Animation attempt, but seems like they don't work here (i.e. the examples don't work)
    fig = plg.Figure(
        data=[plg.Scatter3d(x=coords[0], y=coords[1], z=coords[2],
            mode='markers',
            marker=dict(
                size=3.5,
                color=values[:, 0],
                colorscale='Viridis',
                opacity=0.5
            )
        )],
        layout=plg.Layout(width=600, height=600,
                         updatemenus=[dict(type="buttons",
                                           buttons=[dict(label="Play",
                                                         method="animate",
                                                         args=[None])])]),
        frames=[plg.Frame(
            data=[plg.Scatter3d(x=coords[0], y=coords[1], z=coords[2],
                mode='markers',
                marker=dict(
                    size=3.5,
                    color=values[:, k],
                    colorscale='Viridis',
                    opacity=0.5
                )
            )]) for k in range(100)]
    )

    fig.show()
