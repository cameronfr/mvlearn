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
# http://numerical.recipes/whp/notes/CanonCorrBySVD.pdf
# https://github.com/moskomule/cca.pytorch/blob/master/cca/cca.py

def cca(X, Y):
    # U, S, V = svd(X),   U_x @ S_x.diag() @ V_x.T = x
    U_x, S_x, V_x = X.svd()
    U_y, S_y, V_y = Y.svd()

    U_u, S_u, V_u = (U_x.t() @ U_y).svd()

    A = V_x @ ((1/S_x.view(-1, 1)) * U_u)
    B = V_y @ ((1/S_y.view(-1, 1)) * V_u)

    # first row of A.T and B.T is first canonical variable.
    return A.t(), B.t(), S_u

# takes in two tensors of shape (Rows x Variables)

def ccaEnergy(X, Y):
    return torch.mean(cca(X, Y)[2])

def testCCA():
    X = torch.FloatTensor(np.random.randn(1000,100))
    Y = torch.FloatTensor(np.random.randn(1000,100))

    ccaResult = cca(X, Y)
    Xcomponents = X @ ccaResult[0].t()
    Ycomponents = Y @ ccaResult[1].t()
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
view1Encoder = SimpleEncoder().to(device)
view2Encoder = SimpleEncoder().to(device)

view1Decoder = SimpleDecoder().to(device)
view2Decoder = SimpleDecoder().to(device)

print("Encoder param count ", np.sum([np.prod(s.shape) for s in view1Encoder.parameters()]))

parameters = [view1Encoder.parameters(), view2Encoder.parameters(), view1Decoder.parameters(), view2Decoder.parameters()]
optim = torch.optim.Adam(itertools.chain(*parameters), lr=0.001)

errors = []
latent1Norms = []
latent2Norms = []
latent1Mins = []
latent2Mins = []
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

        view1Error = torch.nn.MSELoss()(view1Decode, view1.to(device))
        view2Error = torch.nn.MSELoss()(view2Decode, view2.to(device))
        corrEnergy = ccaEnergy(view1Latent, view2Latent)
        reconstructionError = view1Error + view2Error
        totalError = reconstructionError -corrEnergy# + torch.abs(1-torch.mean(torch.norm(view1Latent, p=2, dim=1))) + torch.abs(1-torch.mean(torch.norm(view2Latent, p=2, dim=1)))
        totalError.backward()
        if sum([torch.sum(torch.isnan(p.grad)) for p in view2Encoder.parameters()]) > 0:
            raise Exception("Gradient nan")
        elif sum([torch.sum(torch.isnan(p.grad)) for p in view1Encoder.parameters()]) > 0:
            raise Exception("Gradient nan")
        else:
            optim.step()

        latent1Norms.append(torch.mean(torch.norm(view1Latent, p=2, dim=1)))
        latent2Norms.append(torch.mean(torch.norm(view2Latent, p=2, dim=1)))
        latent1Mins.append(torch.min(view1Latent))
        latent2Mins.append(torch.min(view2Latent))

        errors.append(totalError.item())
        # testErrors.append(testError())
        if (idx % 300 == 0):
            print(testError())
            plt.plot(errors)
            # plt.plot(testErrors)
            plotly()
print("done")
plt.plot(latent1Norms[:650])
plt.plot(latent1Mins[:650])

# Test that super lower values aren't causing the NaN gradient
x = torch.randn(10, 10)
x[np.random.randint(10)][np.random.randint(10)] = 1e-14
x.requires_grad = True
total = torch.sum(x.svd()[0]) + torch.sum(x.svd()[1]) + torch.sum(x.svd()[2])
total.backward()
torch.sum(torch.isnan(x.grad))

def testError():
    view1, view2, label = next(iter(testDataloader))
    with torch.no_grad():
        view1Latent = view1Encoder(view1.to(device))
        view2Latent = view2Encoder(view2.to(device))
        view1Decode = view1Decoder(view1Latent)
        view2Decode = view2Decoder(view2Latent)

        view1Error = torch.nn.MSELoss()(view1Decode, view1.to(device))
        view2Error = torch.nn.MSELoss()(view2Decode, view2.to(device))
        corrEnergy = ccaEnergy(view1Latent, view2Latent)
        reconstructionError = view1Error + view2Error
        totalError = 0.01 * reconstructionError - corrEnergy

        totalError = totalError.item()
    return totalError
