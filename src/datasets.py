import os
import numpy as np
import torch
from torch.utils.data import Dataset


class FaceEmbeddingDataset(Dataset):
    def __init__(self, emb_dir):
        self.samples = []
        self.label2idx = {}
        for i, person in enumerate(os.listdir(emb_dir)):
            self.label2idx[person] = i
            person_dir = os.path.join(emb_dir, person)
            for file in os.listdir(person_dir):
                if file.endswith('.npy'):
                    self.samples.append((os.path.join(person_dir, file), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        emb = np.load(path)
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
