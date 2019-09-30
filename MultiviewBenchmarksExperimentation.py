import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as plg
from concurrent.futures import ProcessPoolExecutor
import nibabel as nib
import skvideo.io
import visdom
import pandas as pd
import os
import io
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
import requests
import mne
# vis = visdom.Visdom() # like wisdom but vis DOM ... ... ...

import sklearn.cross_decomposition

from plotly import offline as py
import plotly.tools as tls
py.init_notebook_mode()
plotly = lambda: py.iplot(tls.mpl_to_plotly(plt.gcf())) #usage: put plotly() under plt.hist(...) and run the two lines together

%matplotlib inline
plt.style.use("ggplot")
%config InlineBackend.figure_format = 'svg'

# data from folders in s3://fcp-indi/data/Projects/INDI/MPI-LEMON
# TODO: utilize data availibility tables in Availability_LEMON_Day2_Data in behavorial data folder.
basePath = "/mnt/disks/mindbrainbodydata"
fmriPath = "fmriPreprocessed"
eegPath = "eegPreprocessed"
behavioralPath = "behavorialTests"


subjects = requests.get("https://fcp-indi.s3.amazonaws.com/data/Projects/INDI/MPI-LEMON/MRI_MPILMBB_LEMON/Participants_LEMON.csv")
subjectsInfo = pd.read_csv(io.BytesIO(subjects.content))
subjects = subjectsInfo.values

def loadFMRI(subjectID):
    totalPath = os.path.join(basePath, fmriPath, subjectID, "func/" + subjectID + "_ses-01_task-rest_acq-AP_run-01_MNI2mm.nii.gz")
    totalPath
    fmriData = nib.load(totalPath).get_fdata()
    return fmriData

# Problem: every subject is missing certain EEG channels.
def loadEEG(subjectID):
    subjectID = subjects[5][0]
    folderPath =  os.path.join(basePath, eegPath, subjectID)
    setPath = os.path.join(folderPath, subjectID + "_EC.set") # there's also eyes open data
    patchedSetPath = os.path.join(folderPath, subjectID + "_EC_patched.set")
    loadedSetFile = scipy.io.loadmat(setPath)

    # fix .fdt filename mismatch in .set file
    fdtName = subjectID + "_EC.fdt"
    loadedSetFile['EEG'][0][0][-1][0] = fdtName
    loadedSetFile['EEG'][0][0][15][0] = fdtName
    scipy.io.savemat(patchedSetPath, loadedSetFile, appendmat=False)

    eegDataObj = mne.io.read_raw_eeglab(patchedSetPath, preload=True)
    # every subject is missing some eeg channels, so standardize s.t. the following rows exist:
    channelsToUse = mne.io.read_raw_eeglab(os.path.join(basePath, eegPath, "sub-032301/sub-032301_EC.set"), preload=False).ch_names

    eegDataFrame = eegDataObj.to_data_frame()
    eegDataChannels = eegDataFrame.columns.tolist()
    #TODO: for every channel in channelsToUse, pull the corresponding column from eegDataFrame, in order. With zero columns for ch that don't exist.

    # eegDataObj.to_data_frame().head() # kinda important. Also the original obj file seems to have alot of metadata.
    eegData = eegDataObj.to_data_frame().values
    eegData.shape
    return eegData

def loadBehavorial(subjectID):
    dataPath = os.path.join(basePath, behavioralPath)
    # know that all CSVs found recursively in below folders are test CSVs
    testFolders = ["Cognitive_Test_Battery_LEMON", "Emotion_and_Personality_Test_Battery_LEMON"]
    allCSVArrays = [] #should cache this but it only takes 50ms to load all the csvs

    for testFolder in testFolders:
        for root, dirs, files in os.walk(os.path.join(dataPath, testFolder)):
            CSVs = list(filter(lambda name: ".csv" in name, files))
            for csv in CSVs:
                array = pd.read_csv(os.path.join(root, csv)).values
                allCSVArrays.append(array)

    #currently, ignoring the meaning of the data, just treating it all conglomerated as one view
    # [a.shape for a in allCSVArrays] #can see that each test is missing < a dozen subjects
    totalFields = np.sum([a.shape[1]-1 for a in allCSVArrays]) # -1 because "subjectID" column
    subjectData = np.empty(totalFields, dtype="object")

    currentFieldIdx = 0
    for csvArr in allCSVArrays:
        subjectList = csvArr[:, 0]
        possibleRowIdx = np.where(subjectID == csvArr[:, 0])[0]
        for column in csvArr[:, 1:].transpose():
            if not len(possibleRowIdx) == 0:
                subjectData[currentFieldIdx] = column[possibleRowIdx[0]]
            else:
                subjectData[currentFieldIdx] = None
            currentFieldIdx += 1

    # verification that code is working
    # np.where(allCSVArrays[-1] == subjectID)
    # allCSVArrays[-1][38]
    return subjectData

# in function because using it with ThreadPoolExecutor for faster loading.
def loadSubject(i):
    fmri = loadFMRI(subjects[i, 0])
    fmri = np.mean(fmri, axis=3)
    eeg = loadEEG(subjects[i, 0])
    behav = loadBehavorial(subjects[i, 0])
    return [fmri, eeg, behav]

# Shouldn't be so slow to load when taking mean across time axis
with ProcessPoolExecutor(10) as executor:
    subjectsData = list(executor.map(loadSubject, [0, 1,2,3,4,5,6,7,9,11]))

np.in1d
allFMRIs = np.stack([datas[0] for datas in subjectsData], axis=0)
allEEGs = [np.mean(datas[1], axis=0) for datas in subjectsData]
# need processing that

allEEGs[6].shape
