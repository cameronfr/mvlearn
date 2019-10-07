import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as plg
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import nibabel as nib
import skvideo.io
import visdom
import pandas as pd
import os
from tqdm import tqdm
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
import re
import mne
# vis = visdom.Visdom() # like wisdom but vis DOM ... ... ...

import sklearn.cross_decomposition
import sklearn.linear_model
import sklearn.decomposition

from plotly import offline as py
import plotly.tools as tls
py.init_notebook_mode()
plotly = lambda: py.iplot(tls.mpl_to_plotly(plt.gcf())) #usage: put plotly() under plt.hist(...) and run the two lines together

%matplotlib inline
plt.style.use("ggplot")
%config InlineBackend.figure_format = 'svg'
np.set_printoptions(suppress=True) # don't use scientific [e.g. 5e10] notation

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
    fmriData = nib.load(totalPath).get_fdata()
    return fmriData

# Problem: every subject is missing certain EEG channels.
def loadEEG(subjectID):
    # subjectID = subjects[5][0]
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
    eegDataFrame = eegDataObj.to_data_frame()
    # every subject is missing some eeg channels, so standardize s.t. the following rows exist:
    # eegDataObj.to_data_frame().head() # kinda important. Also the original obj file seems to have alot of metadata.


    channelsToUse = mne.io.read_raw_eeglab(os.path.join(basePath, eegPath, "sub-032301/sub-032301_EC.set"), preload=False).ch_names
    newChannelRows = []
    for channelToUse in channelsToUse:
        if channelToUse in eegDataFrame.columns:
            newChannelRows.append(eegDataFrame[channelToUse].values)
        else:
            newChannelRows.append(np.full(eegDataFrame.shape[0], np.NaN))

    return np.stack(newChannelRows, axis=0)

def loadBehavorial(subjectID):
    dataPath = os.path.join(basePath, behavioralPath)
    # know that all CSVs found recursively in below folders are test CSVs
    testFolders = ["Cognitive_Test_Battery_LEMON", "Emotion_and_Personality_Test_Battery_LEMON"]
    allCSVArrays = [] #should cache this but it only takes 50ms to load all the csvs

    for testFolder in testFolders:
        for root, dirs, files in os.walk(os.path.join(dataPath, testFolder)):
            CSVs = list(filter(lambda name: ".csv" in name, files))
            for csv in CSVs:
                dataFrame = pd.read_csv(os.path.join(root, csv))
                array = dataFrame.values
                allCSVArrays.append(array)

    # currently, ignoring the meaning of the data, just treating it all conglomerated as one view. would be nice to return
    # names for each column, where the name is "csv test name: in-csv column name"
    # [a.shape for a in allCSVArrays] #can see that each test is missing data for  < a dozen subjects
    totalFields = np.sum([a.shape[1]-1 for a in allCSVArrays]) # -1 because "subjectID" column
    subjectData = np.empty(totalFields)
    # [csv[:, -1] for csv in allCSVArrays] # not all the last columns are "comments"
    # [csv[:, 0] for csv in allCSVArrays]  # all the first columns are the subject id

    currentFieldIdx = 0
    for csvArr in allCSVArrays:
        subjectList = csvArr[:, 0]
        possibleRowIdx = np.where(subjectID == csvArr[:, 0])[0]
        for column in csvArr[:, 1:].transpose():
            if not len(possibleRowIdx) == 0:
                fieldValue = column[possibleRowIdx[0]]
                fieldValue = pd.to_numeric(re.sub("<|>", "", str(fieldValue)), errors="coerce")
                subjectData[currentFieldIdx] = fieldValue
            else:
                subjectData[currentFieldIdx] = np.nan
            currentFieldIdx += 1

    # verification that code is working
    # np.where(allCSVArrays[-1] == subjectID)
    # allCSVArrays[-1][38]
    return subjectData

# in function because using it with ThreadPoolExecutor for faster loading.
def loadSubjectMeanOverTime(i):
    try:
        fmri = loadFMRI(subjects[i, 0])
        fmri = np.mean(fmri, axis=3)
        eeg = loadEEG(subjects[i, 0])
        eeg = np.mean(eeg, axis=1)
        behav = loadBehavorial(subjects[i, 0])
        return [fmri, eeg, behav]
    except Exception as e:
        print("Failed to load subject {}, {}, error {}".format(i, subjects[i,0], e))
        return None

def loadSubject(i):
    try:
        fmri = loadFMRI(subjects[i, 0])
        eeg = loadEEG(subjects[i, 0])
        behav = loadBehavorial(subjects[i, 0])
        return [fmri, eeg, behav]
    except Exception as e:
        print("Failed to load subject {}, {}, error {}".format(i, subjects[i,0], e))
        return None


#------------------------------ BENCHMARKING PCA / CCA BASIC IDEAS ------------------------------#

# Shouldn't be so slow to load when taking mean across time axis
with ProcessPoolExecutor(10) as executor:
    subjectsToLoad = range(30)
    subjectsDataAll = executor.map(loadSubject, subjectsToLoad)
    subjectsData = list(filter(lambda x: not x == None, list(subjectsDataAll)))

# each index of subjectsData is full data for a view
subjectsData = [np.stack([s[i] for s in subjectsData]) for i in range(len(subjectsData[0]))]
import pickle
for i in range(3):
    subjectsData[i] = subjectsData[i].reshape(subjectsData[i].shape[0], -1)

# with open("first30subjectsDataMeanOverTime", "wb") as f:
#     pickle.dump(subjectsData, f)

viewFMRI = np.copy(subjectsData[0])
viewEEG = np.copy(subjectsData[1])
viewEEG[np.isnan(viewEEG)] = 0 # 6 nans per 10 subj
viewBehav = np.copy(subjectsData[2])
viewBehav[np.isnan(viewBehav)] = 0 # 247 nans per 10 subj

# normalize features of the thing to be predicted
viewBehav = (viewBehav - np.mean(viewBehav, axis=0).reshape(1, -1)) / np.std(viewBehav, axis=0).reshape(1, -1)
viewBehav[np.isnan(viewBehav)] = 0 # 247 nans per 10 subj


# random k-fold, seeing what L2 error of linear regression is on
# (CCA(FMRI, EEG) latents (2 components from each)) -> viewBehav    AVG L2: 1.22
# PCA(EEG), PCA(FMRI) (2 components from each) -> viewBehav         AVG L2: 1.77
# random noise, (4 components) -> viewBehav
# not really sure what is going on here, but this is a way to benchmark any black box latents thing.

# nullDistErrors = []
# for i in range(1000):
latentType = "CCA"
# latentType = "NOISE"
# latentType = "PCA"
if (latentType == "CCA"):
    latents1, latents2 = sklearn.cross_decomposition.CCA(n_components=2).fit_transform(viewFMRI, viewEEG)
    latents = np.concatenate([latents1, latents2], axis=1)
elif (latentType == "PCA"):
    latents1 = sklearn.decomposition.PCA(n_components=2).fit_transform(viewFMRI)
    latents2 = sklearn.decomposition.PCA(n_components=2).fit_transform(viewEEG)
    latents = np.concatenate([latents1, latents2], axis=1)
elif (latentType == "NOISE"):
    latents = np.random.randn(viewFMRI.shape[0], 4)
errors = []
for i in range(100):
    subjCount = viewFMRI.shape[0]
    heldoutIndexes = np.random.choice(np.arange(subjCount), 10)
    heldoutMask = np.zeros(subjCount, dtype=np.bool)
    heldoutMask[heldoutIndexes] = 1
    model = sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=True)
    model.fit(latents[~heldoutMask], viewBehav[~heldoutMask])
    predictions = model.predict(latents[heldoutMask])
    L2error = np.mean(np.power(viewBehav[heldoutMask] - predictions, 2))
    errors.append(L2error)
np.mean(errors)
# nullDistErrors.append(np.mean(errors))

plt.hist(nullDistErrors, bins=20)
np.searchsorted(np.sort(nullDistErrors), 1.22) / len(nullDistErrors) # p-value 0.1
np.searchsorted(np.sort(nullDistErrors), 1.77) / len(nullDistErrors) # p-value 0.1
len(nullDistErrors)


#------------------------------ FMRI 3D Conv Autoencoder ------------------------------#

import torch
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import itertools

# for column in subjectsData[2].transpose()[-10:]:
#     plt.hist(column[~np.isnan(column)], bins=20)
#     plt.show()

# Shouldn't be so slow to load when taking mean across time axis
with ThreadPoolExecutor(10) as executor:
    subjectsToLoad = range(10)
    subjectsDataAll = executor.map(loadSubject, subjectsToLoad)
    subjectsData = list(filter(lambda x: not x == None, list(subjectsDataAll)))

trainingSubjects = subjectsData[:7]
testSubjects = subjectsData[7:]

class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, subjects):
        super().__init__()
        self.subjects = subjects

    def __len__(self):
        return np.sum([s[0].shape[-1] for s in self.subjects])

    def __getitem__(self, x):
        slice = self.getRandom3DFmriSlice()
        slice = (slice - np.mean(slice)) / np.std(slice)
        return torch.FloatTensor(slice)

    def getRandom3DFmriSlice(self):
        subjectIndex = np.random.randint(len(self.subjects))
        subject = self.subjects[subjectIndex]
        fmri = subject[0]
        timeIndex = np.random.randint(fmri.shape[-1])
        randomSlice = fmri[:, :, :, timeIndex]
        return randomSlice

dataset = SliceDataset(trainingSubjects)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 4, 3, stride=2)
        self.conv2 = torch.nn.Conv3d(4, 8, 3, stride=2)
        self.conv3 = torch.nn.Conv3d(8, 16, 3, stride=2)
        self.conv4 = torch.nn.Conv3d(16, 32, 3, stride=2)
        self.conv5 = torch.nn.Conv3d(32, 64, 3, stride=2)
        self.conv6 = torch.nn.Conv3d(64, 64, (1, 2, 1), stride=2)

    def forward(self, x):
        # relu = torch.nn.ReLU()
        relu = torch.nn.LeakyReLU()
        out = x.unsqueeze(1) # create channel dimension for the 3d data
        out = relu(self.conv1(out))
        out = relu(self.conv2(out))
        out = relu(self.conv3(out))
        out = relu(self.conv4(out))
        out = relu(self.conv5(out))
        out = torch.nn.Tanh()(self.conv6(out))
        out = out.squeeze()
        return out

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rconv1 = torch.nn.ConvTranspose3d(64, 64, (1,2,1), stride=2)
        self.rconv2 = torch.nn.ConvTranspose3d(64, 32, 3, stride=2, output_padding=(1, 0, 1))
        self.rconv3 = torch.nn.ConvTranspose3d(32, 16, 3, stride=2, output_padding=1)
        self.rconv4 = torch.nn.ConvTranspose3d(16, 8, 3, stride=2, output_padding=1)
        self.rconv5 = torch.nn.ConvTranspose3d(8, 4, 3, stride=2, output_padding=(0, 1, 0))
        self.rconv6 = torch.nn.ConvTranspose3d(4, 1, 3, stride=2)

    def forward(self, x):
        # relu = torch.nn.ReLU()
        relu = torch.nn.LeakyReLU()
        out = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = relu(self.rconv1(out))
        out = relu(self.rconv2(out))
        out = relu(self.rconv3(out))
        out = relu(self.rconv4(out))
        out = relu(self.rconv5(out))
        out = self.rconv6(out)
        return out.squeeze()


encoder = Encoder().to(device)
decoder = Decoder().to(device)
print("enc params", np.sum([np.prod(p.shape) for p in encoder.parameters()]))
print("dec params", np.sum([np.prod(p.shape) for p in decoder.parameters()]))

encoderOptim = torch.optim.Adam(encoder.parameters(), lr=0.05)
decoderOptim = torch.optim.Adam(decoder.parameters(), lr=0.05)

errors = []
for i in range(500):
    # for batch in dataloader:
        encoderOptim.zero_grad()
        decoderOptim.zero_grad()
        latents = encoder(batch.to(device))
        reconstruction = decoder(latents)
        reconstruction.shape
        batch.shape
        error = torch.mean(torch.pow(reconstruction - batch.to(device).detach(), 2))
        error.backward()
        encoderOptim.step()
        decoderOptim.step()
        errors.append(error.item())
        print("error", error.item())

plt.plot(errors)
plt.grid(False)
plt.imshow(torch.cat([batch[0][:, :, 30], reconstruction.cpu().detach()[0][:, :, 30]], dim=1))
