import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

import matplotlib.pyplot as plt
import numpy as np

plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)


x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())


from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader



# class Mnist_CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

#     def forward(self, xb):
#         xb = xb.view(-1, 1, 28, 28)
#         xb = F.relu(self.conv1(xb))
#         xb = F.relu(self.conv2(xb))
#         xb = F.relu(self.conv3(xb))
#         xb = F.avg_pool2d(xb, 4)
#         return xb.view(-1, xb.size(1))

# model = Mnist_CNN()

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

def preprocess(x, y):
    return x.view(-1,1,28,28), y

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func
    def __len__(self):
        return len(self.dl)
    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

batch_size = 128
train_ds = TensorDataset(x_train, y_train)
# train_dl = DataLoader(train_ds, batch_size, True)
valid_ds = TensorDataset(x_valid, y_valid)
# valid_dl = DataLoader(valid_ds, batch_size)
train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)


model = nn.Sequential(
    # Lambda(preprocess),
    nn.Conv2d(1, 16, 3, 2, 1),
    nn.ReLU(),
    nn.Conv2d(16, 16, 3, 2, 1),
    nn.ReLU(),
    nn.Conv2d(16, 16, 3, 2, 1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

from torch import optim
loss_func = F.cross_entropy
optim = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    val_loss_list = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        model.eval()
        with torch.no_grad():
            for xb, yb in valid_dl:
                val_loss = loss_func(model(xb), yb)
                val_loss_list.append(val_loss)

        print(epoch, val_loss)
    return val_loss_list

val_loss_list = fit(10, model, loss_func, optim, train_dl, valid_dl)

print(val_loss_list)
