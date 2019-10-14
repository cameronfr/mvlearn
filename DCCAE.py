import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import PIL
import numpy as np
import itertools
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from plotly import offline as py
import plotly.tools as tls
py.init_notebook_mode()
plotly = lambda: py.iplot(tls.mpl_to_plotly(plt.gcf())) #usage: put plotly() under plt.hist(...) and run the two lines together

%matplotlib inline
plt.style.use("ggplot")
%config InlineBackend.figure_format = 'svg'
np.set_printoptions(suppress=True) # don't use scientific [e.g. 5e10] notation


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

encoder = SimpleEncoder().to(device)
view1Decoder = SimpleDecoder().to(device)
view2Decoder = SimpleDecoder().to(device)
# encoder.load_state_dict(torch.load("encoderMNISTDCCAE"))

# a hefty 2.9 mil params... normal conv2d mnist has <50k
print("Encoder param count ", np.sum([np.prod(s.shape) for s in encoder.parameters()]))

parameters = [encoder.parameters(), view1Decoder.parameters(), view2Decoder.parameters()]
optim = torch.optim.Adam(itertools.chain(*parameters), lr=0.001)

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

torch.save(encoder.state_dict(), "encoderMNISTDCCAE")

plt.imshow(dataset[0][0].squeeze())
plt.imshow(view1[0][0])
plt.imshow(view1Decode.detach().cpu()[4][0])
view1.shape

plt.imshow(torchvision.utils.make_grid(view1Decode[:16]).detach().cpu().numpy().transpose(1, 2, 0))
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
view1.shape

tsneEmbeddingsNoEncode = tsne.fit_transform(view1.view(-1, 784).numpy())
tsneEmbeddingsNoEncodeNoisy = tsne.fit_transform(view2.view(-1, 784).numpy())
plt.scatter(*tsneEmbeddings.transpose(), c=pointColors, s=5)
plotly()
plt.scatter(*tsneEmbeddingsNoEncode.transpose(), c=pointColors, s=5)
plt.scatter(*tsneEmbeddingsNoEncodeNoisy.transpose(), c=pointColors, s=5)
