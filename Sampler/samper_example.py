import torch
from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        return sample


x = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
x = np.array(x)

dataset = TestDataset(x)

# print normal order
[print(d) for i, d in enumerate(dataset)]
print()

# print using RandomSampler
[print(d) for d in RandomSampler(dataset.data)]
# => shuffle 된 index가 출력됨

# MelanomaSampler 정의 (2class imbalance case)
class MelanomaSampler(Sampler):
    def __init__(self, dataset, malignant_pct=0.5):
        self.dataset = dataset
        self.n = len(dataset)
        self.malignant_pct = malignant_pct

    def __iter__(self):
        malignant_idxs = np.where(self.dataset.data==1)[0]
        benign_idxs = np.where(self.dataset.data==0)[0]
        malignant = np.random.choice(malignant_idxs, int(self.n * self.malignant_pct), replace=True)
        benign = np.random.choice(benign_idxs, int(self.n * (1-self.malignant_pct))+1, replace=True)
        idxs = np.hstack([malignant, benign])
        np.random.shuffle(idxs)
        # idxs = idxs[:n]
        return iter(idxs)

    def __len__(self):
        return self.n

train_dl_w_custom_sampler = DataLoader(dataset=dataset, batch_size=4, sampler=MelanomaSampler(dataset)) # num_workers=1, 

for i, d in enumerate(train_dl_w_custom_sampler):
    print(d)

print()