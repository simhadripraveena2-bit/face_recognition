import os
import numpy as np
import torch
from torch.utils.data import Dataset


class FaceEmbeddingDataset(Dataset):
    def __init__(self, emb_dir):
        self.samples = []
        self.label2idx = {}
        for i, person in enumerate(os.listdir(emb_dir)):
            person_dir = os.path.join(emb_dir, person)
            if not os.path.isdir(person_dir):
                continue
            self.label2idx[person] = i
            for file in os.listdir(person_dir):
                if file.endswith('.npy') and not file.endswith('.geo.npy'):
                    self.samples.append((os.path.join(person_dir, file), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emb_path, label = self.samples[idx]
        emb = np.load(emb_path).astype(np.float32).reshape(-1)
        geo_path = emb_path.replace('.npy', '.geo.npy')
        if os.path.exists(geo_path):
            geo = np.load(geo_path).astype(np.float32).reshape(-1)
        else:
            geo = np.zeros(5, dtype=np.float32)
        combined = np.concatenate([emb, geo], axis=0)
        return torch.tensor(combined, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
