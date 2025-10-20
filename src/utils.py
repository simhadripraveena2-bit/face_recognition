import os
import numpy as np

def save_embedding(embedding, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embedding)

def load_embedding(path):
    return np.load(path)
