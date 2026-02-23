import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=DEVICE)


def get_aligned_face_tensor(image: Image.Image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    face = mtcnn(np.array(image))
    if face is None:
        return None
    return face.unsqueeze(0).to(DEVICE)
