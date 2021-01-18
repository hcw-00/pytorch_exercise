from torchvision.datasets import CIFAR10
from torchvision import transforms
import tempfile
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

print(tempfile.gettempdir())

cifar10 = CIFAR10(tempfile.gettempdir(),
                  train=True,
                  download=True)

_mean = cifar10.data.mean(axis=(0,1,2)) / 255
_std = cifar10.data.std(axis=(0,1,2)) / 255

class dataset_1(Dataset):
    def __init__(self, xy, xy2):
        print()
        self.xy = xy
        self.xy2 = xy2
        self.len = len(self.xy)
        
    def __getitem__(self, idx):
        print()
        x, y = self.xy[idx]
        x2, y2 = self.xy2[idx]
        x, x2 = np.asarray(x), np.asarray(x2)
        x = x - x2
        x, x2 = Image.fromarray(x), Image.fromarray(x2)
        # x = transforms.Resize(size=(100,100))(x)
        return (x,y)
    def __len__(self):
        print()
        return self.len
        


print()

dataset1 = dataset_1(cifar10, cifar10)
dataset1[0][0].show()
cifar10[0][0].show()
print()