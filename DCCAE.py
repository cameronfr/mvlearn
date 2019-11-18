import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import PIL
import numpy as np
import itertools
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sklearn.cross_decomposition

from plotly import offline as py
import plotly.tools as tls
py.init_notebook_mode()
plotly = lambda: py.iplot(tls.mpl_to_plotly(plt.gcf())) #usage: put plotly() under plt.hist(...) and run the two lines together

%matplotlib inline
plt.style.use("ggplot")
%config InlineBackend.figure_format = 'svg'
np.set_printoptions(suppress=True) # don't use scientific [e.g. 5e10] notation

# %%

class NoisyMnist(Dataset):

    MNIST_MEAN, MNIST_STD = (0.1307, 0.3081)

    def __init__(self, train=True):
        super().__init__()
        self.mnistDataset = datasets.MNIST("./mnist", train=train, download=True)

    def __len__(self):
        return len(self.mnistDataset)

    def __getitem__(self, idx):
        randomIndex = lambda: np.random.randint(len(self.mnistDataset))
        image1, label1 = self.mnistDataset[idx]
        image2, label2 = self.mnistDataset[randomIndex()]
        while not label1 == label2:
            image2, label2 = self.mnistDataset[randomIndex()]

        image1 = torchvision.transforms.RandomRotation((-45, 45), resample=PIL.Image.BICUBIC)(image1)
        image2 = torchvision.transforms.RandomRotation((-45, 45), resample=PIL.Image.BICUBIC)(image2)
        image1 = np.array(image1) / 255
        image2 = np.array(image2) / 255

        image2 = np.clip(image2 + np.random.uniform(0, 1, size=image2.shape), 0, 1)

        image1 = (image1 - self.MNIST_MEAN) / self.MNIST_STD
        image2 = (image2 - self.MNIST_MEAN) / self.MNIST_STD

        image1 = torch.FloatTensor(image1).unsqueeze(0)
        image2 = torch.FloatTensor(image2).unsqueeze(0)

        return (image1, image2, label1)

class SimpleEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(784, 1024)
        self.layer2 = torch.nn.Linear(1024, 1024)
        self.layer3 = torch.nn.Linear(1024, 1024)
        self.layer4 = torch.nn.Linear(1024, 20)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.nn.Sigmoid()(self.layer1(x))
        x = torch.nn.Sigmoid()(self.layer2(x))
        x = torch.nn.Sigmoid()(self.layer3(x))
        x = torch.nn.Sigmoid()(self.layer4(x))
        return x

class SimpleDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(20, 1024)
        self.layer2 = torch.nn.Linear(1024, 1024)
        self.layer3 = torch.nn.Linear(1024, 1024)
        self.layer4 = torch.nn.Linear(1024, 784)

    def forward(self, x):
        x = torch.nn.Sigmoid()(self.layer1(x))
        x = torch.nn.Sigmoid()(self.layer2(x))
        x = torch.nn.Sigmoid()(self.layer3(x))
        x = self.layer4(x)
        x = x.view(-1, 28, 28).unsqueeze(1)
        return x



# their SplitAE (768-> 1024, 1024-> 1024, 1024->1024, 1024 -> L), L = 5 or 10 or 15 or 20 or 50, paper ambig.
# should replicate graph and then compare to convolutional version with same amt of parameters.
dataset = NoisyMnist(train=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
testDataset = NoisyMnist(train=False)
testDataloader = DataLoader(testDataset, batch_size=128, shuffle=True, num_workers=8)

# check out dataset we made
plt.imshow(torch.cat(dataset[np.random.randint(len(dataset))][:2], dim=2).squeeze().numpy(), cmap="gray")

# %%
# Implement splitAE

encoder = SimpleEncoder().to(device)
view1Decoder = SimpleDecoder().to(device)
view2Decoder = SimpleDecoder().to(device)
# encoder.load_state_dict(torch.load("encoderMNISTDCCAE"))

# a hefty 2.9 mil params... normal conv2d mnist has <50k
print("Encoder param count ", np.sum([np.prod(s.shape) for s in encoder.parameters()]))

parameters = [encoder.parameters(), view1Decoder.parameters(), view2Decoder.parameters()]
optim = torch.optim.Adam(itertools.chain(*parameters), lr=0.0001)

errors = []
testErrors = []
for epoch in range(20):
    for idx, (view1, view2, label) in enumerate(dataloader):
        optim.zero_grad()

        latent = encoder(view1.to(device))
        view1Decode = view1Decoder(latent)
        view2Decode = view2Decoder(latent)
        view1Error = torch.nn.MSELoss()(view1Decode, view1.to(device))
        view2Error = torch.nn.MSELoss()(view2Decode, view2.to(device))
        totalError = view1Error + view2Error
        totalError.backward()
        optim.step()

        errors.append(totalError.item())
        # testErrors.append(testError())
        if (idx % 300 == 0):
            print(testError())
            plt.plot(errors)
            # plt.plot(testErrors)
            plotly()
print("done")

def testError():
    view1, view2, label = next(iter(testDataloader))
    with torch.no_grad():
        latent = encoder(view1.to(device))
        view1Decode = view1Decoder(latent)
        view2Decode = view2Decoder(latent)
        view1Error = torch.nn.MSELoss()(view1Decode, view1.to(device))
        view2Error = torch.nn.MSELoss()(view2Decode, view2.to(device))
        totalError = view1Error + view2Error
        totalError = totalError.item()
    return totalError

torch.save([encoder.state_dict(), view1Decoder.state_dict(), view2Decoder.state_dict(), optim.state_dict()], "AELongTrainMNISTDCCAE")

plt.grid(False)
plt.imshow(torch.cat(tuple(view1[:8]), dim=2).squeeze().detach().cpu())
plt.imshow(torch.cat(tuple(view1Decode[:8]), dim=2).squeeze().detach().cpu())
# most of the error is coming from the view2 reconstruction -- so currently this SplitAE seems (would have to test) equivalent to just an autoencdoer on view1
plt.imshow(torch.cat(tuple(view2Decode[:8]), dim=2).squeeze().detach().cpu())
plt.show()

plt.imshow(torchvision.utils.make_grid(view1Decode[:1]).detach().cpu().numpy().transpose(1, 2, 0))
plt.hist(view1.numpy().ravel())

# Trying to replicate the T-SNE figure of SplitAE
from MulticoreTSNE import MulticoreTSNE as TSNE #sklearn TSNE too slow

testDataset = NoisyMnist(train=False)
testDataloader = DataLoader(testDataset, batch_size=10000, shuffle=True, num_workers=8)
with torch.no_grad():
    view1, view2, labels = next(iter(testDataloader))
    latents = encoder(view1.to(device))
latents.shape

pointColors = []
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
origColors = [[55, 55, 55], [255, 34, 34], [38, 255, 38], [10, 10, 255], [255, 12, 255], [250, 200, 160], [120, 210, 180], [150, 180, 205], [210, 160, 210], [190, 190, 110]]
origColors = (np.array(origColors)) / 255
for l in labels.cpu().numpy():
    pointColors.append(tuple(origColors[l].tolist()))

tsne = TSNE(n_jobs=12)
tsneEmbeddings = tsne.fit_transform(latents.cpu().numpy())

tsneEmbeddingsNoEncode = tsne.fit_transform(view1.view(-1, 784).numpy())
tsneEmbeddingsNoEncodeNoisy = tsne.fit_transform(view2.view(-1, 784).numpy())
plt.scatter(*tsneEmbeddings.transpose(), c=pointColors, s=5)
plotly()
plt.scatter(*tsneEmbeddingsNoEncode.transpose(), c=pointColors, s=5)
plt.scatter(*tsneEmbeddingsNoEncodeNoisy.transpose(), c=pointColors, s=5)

# %%
# CCA experimentation

data = np.random.randn(2,10)
np.sum(np.power(data[0] - np.mean(data[0]), 2)) / data[0].shape

np.sum((data[0] - np.mean(data[0])) * (data[1]-np.mean(data[1]))) / data[0].shape

np.cov(data, ddof=0)
np.mean(np.power(data[0] - np.mean(data[0]), 2))
np.var(data[0], ddof=0)


# %%
# Implement DCCAE


# Below is CCA formula given by
# http://numerical.recipes/whp/notes/CanonCorrBySVD.pdf
# https://github.com/moskomule/cca.pytorch/blob/master/cca/cca.py
# backwards pass doesn't work when X or Y non singular
def cca_svd(Xin, Yin):
    # U, S, V = svd(X),   U_x @ S_x.diag() @ V_x.T = x
    X = Xin - torch.mean(Xin)
    Y = Yin - torch.mean(Yin)
    U_x, S_x, V_x = X.svd()
    U_y, S_y, V_y = Y.svd()

    U_u, S_u, V_u = (U_x.t() @ U_y).svd()

    A = V_x @ ((1/S_x.view(-1, 1)) * U_u)
    B = V_y @ ((1/S_y.view(-1, 1)) * V_u)

    # first row of A.T and B.T is first canonical variable.
    return A.t(), B.t(), S_u

# Below is CCA formula given by
# https://ttic.uchicago.edu/~klivescu/papers/andrew_icml2013.pdf
# Gives opportunity to prevent backprop failure (via adding identity) when input singular.
def cca(Xin, Yin, regularizationλ=0):
    # Xin = view1Latent.detach()
    # Yin = view2Latent.detach()
    # Xin.requires_grad = True
    # Yin.requires_grad = True
    # regularizationλ = 0.5

    X = Xin - torch.mean(Xin, axis=0)
    Y = Yin - torch.mean(Yin, axis=0)
    k = min(Xin.shape[1], Yin.shape[1])
    covXX = (X.t() @ X) + regularizationλ*torch.eye(X.shape[1], device=X.device)
    covYY = (Y.t() @ Y) + regularizationλ*torch.eye(Y.shape[1], device=X.device)
    covXY = (X.t() @ Y)

    U_x, S_x, V_x = covXX.svd()
    U_y, S_y, V_y = covYY.svd()
    covXXinvHalf = V_x @ (S_x.sqrt().reciprocal().diag()) @ U_x.t()
    covYYinvHalf = V_y @ (S_y.sqrt().reciprocal().diag()) @ U_y.t()
    T = covXXinvHalf @ covXY @ covYYinvHalf
    U, S, V = T.svd()
    A = covXXinvHalf @ U[:, :k]
    B = covYYinvHalf @ V[:, :k]
    return A.t(), B.t(), S
    # s = S.mean()
    # s.backward()
    # torch.isnan(Yin.grad).sum()

# takes in two tensors of shape (Rows x Variables)
def ccaEnergy(X, Y, regularizationλ=0):
    return torch.mean(cca(X, Y, regularizationλ)[2])

def testCCA():
    X = torch.FloatTensor(np.random.randn(1000,100)) + 20
    Y = torch.FloatTensor(np.random.randn(1000,200)) + 20

    ccaResult = cca(X, Y)
    Xcomponents = (X - X.mean()) @ ccaResult[0].t()
    Ycomponents = (Y - Y.mean()) @ ccaResult[1].t()
    canonicalCorrelations = ccaResult[2]

    # significantly worse, on 10,000samples x 1000 variables randn on mbp, after 2min hits "max iter error", vs 1.44s for above func.
    sklearnCCAComponents = sklearn.cross_decomposition.CCA(n_components=X.shape[1]).fit_transform(X, Y)
    sklearnXcomponents = torch.FloatTensor(sklearnCCAComponents[0])
    sklearnYcomponents = torch.FloatTensor(sklearnCCAComponents[1])

    def correlation(xVec, yVec):
        covariance = torch.mean((xVec - xVec.mean()) * (yVec - yVec.mean()))
        stdprod = (torch.std(xVec, unbiased=False) * torch.std(yVec, unbiased=False))
        corr = covariance / stdprod
        return corr

    withinAlgoDiffs = []
    sklearnDiffs = []
    for i in range(canonicalCorrelations.shape[0]):
        corr = correlation(Xcomponents[:, i], Ycomponents[:, i])
        sklearnCorr = correlation(sklearnXcomponents[:, i], sklearnYcomponents[:, i])
        withinAlgoDiffs.append((torch.abs(corr - canonicalCorrelations[i])).item())
        sklearnDiffs.append((torch.abs(corr - sklearnCorr)).item())
        assert withinAlgoDiffs[-1] <= 5e-2
        assert sklearnDiffs[-1] <= 5e-2
    plt.hist(withinAlgoDiffs)
    plt.hist(sklearnDiffs)

    # make sure uncorrelated canonical variables when i \neq j
    # also tensorized version of the withinAlgoDiff check
    covariances = ((Xcomponents.t() - Xcomponents.mean()) @ (Ycomponents.t().t() - Ycomponents.mean())) / Xcomponents.t().shape[1]
    stdProducts = torch.std(Ycomponents.t(), dim=1).view(1, -1) * torch.std(Xcomponents.t(), dim=1).view(-1, 1)
    correlations = covariances / stdProducts
    assert torch.all(correlations.diagonal() - canonicalCorrelations < 5e-2)
    assert torch.all(correlations - correlations.diagonal().diag() < 5e-2)
    # diagonal() gets the diagional, x.diag() makes square zero matrix with x.diagonal() = x

testCCA() # svd not working on V100 machine for pytorch >1.1 (1.1 works). Mb need to upgrade CUDA.

# SVD implementation has gradient for singular inputs where rows not same but non linindep at all??
# X = torch.stack([torch.randn(10)]* 20)
# Y = torch.stack([torch.randn(15)]* 20)
# X.shape
# X.requires_grad = True
# sdum = torch.sum(cca(X, Y, regularizationλ=0)[2])
# sdum.backward()
# X.grad

# Notes: when set learning rate too high (e.g. 0.01 instead of 0.001) , the Andrew et al CCA method with regularization still has a NaN gradient.
# Other SVD cca reaches NaN gradient with LR 0.001.
# When set regularization parameter higher (e.g. 0.5 instead of 0.0001), it takes longer before a NaN gradient is reached.
# Should implement manually derived gradients of DCCA and see if that helps.

view1Encoder = SimpleEncoder().to(device)
view2Encoder = SimpleEncoder().to(device)

view1Decoder = SimpleDecoder().to(device)
view2Decoder = SimpleDecoder().to(device)

# savedStates = torch.load("batch24000bs128_DCCAE_MNIST.pytorch")
# view1Encoder.load_state_dict(savedStates[0])
# view2Encoder.load_state_dict(savedStates[1])
# view1Decoder.load_state_dict(savedStates[2])
# view2Decoder.load_state_dict(savedStates[3])

print("Encoder param count ", np.sum([np.prod(s.shape) for s in view1Encoder.parameters()]))

parameters = [view1Encoder.parameters(), view2Encoder.parameters(), view1Decoder.parameters(), view2Decoder.parameters()]
optim = torch.optim.Adam(itertools.chain(*parameters), lr=0.001)

errors = []
testErrors = []
for epoch in range(20):
    for idx, (view1, view2, label) in enumerate(dataloader):
        optim.zero_grad()

        view1Latent = view1Encoder(view1.to(device))
        view2Latent = view2Encoder(view2.to(device))
        # if (torch.sum(torch.isnan(view2Latent)) > 0):
        #     print(torch.sum(torch.isnan(view2Latent)))
        view1Decode = view1Decoder(view1Latent)
        view2Decode = view2Decoder(view2Latent)
        plt.imshow(view2[1].squeeze().numpy())
        plt.imshow(view2Decode[1].cpu().detach().squeeze().numpy())

        view1Error = torch.nn.MSELoss()(view1Decode, view1.to(device))
        view2Error = torch.nn.MSELoss()(view2Decode, view2.to(device))
        corrEnergy = ccaEnergy(view1Latent, view2Latent, regularizationλ=10)
        reconstructionError = view1Error + view2Error
        totalError = reconstructionError - corrEnergy
        totalError.backward()
        if sum([torch.sum(torch.isnan(p.grad)) for p in view2Encoder.parameters()]) > 0:
            raise Exception("Gradient nan")
        elif sum([torch.sum(torch.isnan(p.grad)) for p in view1Encoder.parameters()]) > 0:
            raise Exception("Gradient nan")
        else:
            optim.step()

        errors.append(totalError.item())
        # testErrors.append(testError())
        if (idx % 300 == 0):
            print(testError())
            plt.plot(errors)
            # plt.plot(testErrors)
            plt.show()
            # plotly()

# torch.save([view1Encoder.state_dict(), view2Encoder.state_dict(), view1Decoder.state_dict(), view2Decoder.state_dict()], "")


# Test that super lower values aren't causing the NaN gradient
# x = torch.randn(10, 10)
# x[np.random.randint(10)][np.random.randint(10)] = 1e-14
# x.requires_grad = True
# total = torch.sum(x.svd()[0]) + torch.sum(x.svd()[1]) + torch.sum(x.svd()[2])
# total.backward()
# torch.sum(torch.isnan(x.grad))

def testError():
    view1, view2, label = next(iter(testDataloader))
    with torch.no_grad():
        view1Latent = view1Encoder(view1.to(device))
        view2Latent = view2Encoder(view2.to(device))
        view1Decode = view1Decoder(view1Latent)
        view2Decode = view2Decoder(view2Latent)

        view1Error = torch.nn.MSELoss()(view1Decode, view1.to(device))
        view2Error = torch.nn.MSELoss()(view2Decode, view2.to(device))
        corrEnergy = ccaEnergy(view1Latent, view2Latent, regularizationλ=0.5)
        reconstructionError = view1Error + view2Error
        totalError = reconstructionError - corrEnergy

        totalError = totalError.item()
    return totalError


# Trying to replicate the T-SNE figure of DCCAE
from MulticoreTSNE import MulticoreTSNE as TSNE #sklearn TSNE too slow

testDataset = NoisyMnist(train=False)
testDataloader = DataLoader(testDataset, batch_size=10000, shuffle=True, num_workers=8)
with torch.no_grad():
    view1, view2, labels = next(iter(testDataloader))
    latents = view1Encoder(view1.to(device))
latents.shape

pointColors = []
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
origColors = [[55, 55, 55], [255, 34, 34], [38, 255, 38], [10, 10, 255], [255, 12, 255], [250, 200, 160], [120, 210, 180], [150, 180, 205], [210, 160, 210], [190, 190, 110]]
origColors = (np.array(origColors)) / 255
for l in labels.cpu().numpy():
    pointColors.append(tuple(origColors[l].tolist()))

tsne = TSNE(n_jobs=12)
tsneEmbeddings = tsne.fit_transform(latents.cpu().numpy())
# tsneEmbeddingsNoEncode = tsne.fit_transform(view1.view(-1, 784).numpy())
# tsneEmbeddingsNoEncodeNoisy = tsne.fit_transform(view2.view(-1, 784).numpy())
plt.xlabel("t-sne component 1")
plt.ylabel("t-sne component 2")
plt.title("DCCAE Noisy MNIST Embedding")
plt.scatter(*tsneEmbeddings.transpose(), c=pointColors, s=5)
plotly()
# plt.scatter(*tsneEmbeddingsNoEncode.transpose(), c=pointColors, s=5)
# plt.scatter(*tsneEmbeddingsNoEncodeNoisy.transpose(), c=pointColors, s=5)


# bigLatent = torch.stack(20*[torch.floor(latent)], dim=0)
# bigLatent.requires_grad = True
#
# energy = torch.sum(bigLatent.svd()[2])
# energy.backward()
# # bigLatent = torch.randn(20,20)
# u, s, v = bigLatent.svd()
# plt.hist(v.detach().numpy().ravel())
# s
# # SVD still working but grad is NaN.
# # diff = bigLatent - (u @ s.diag() @ v.t())
# # torch.max(torch.abs(diff))
# bigLatent.grad
