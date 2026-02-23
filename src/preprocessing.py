import os

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from src.face_pipeline import get_aligned_face_tensor


def crop_faces(input_dir, output_dir):
    """Detect/aligned faces using the shared pipeline used by inference."""
    os.makedirs(output_dir, exist_ok=True)
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        if not os.path.isdir(person_dir):
            continue
        out_person_dir = os.path.join(output_dir, person)
        os.makedirs(out_person_dir, exist_ok=True)

        for img_file in tqdm(os.listdir(person_dir), desc=person):
            img_path = os.path.join(person_dir, img_file)
            if not os.path.isfile(img_path):
                continue
            image = Image.open(img_path).convert('RGB')
            face = get_aligned_face_tensor(image)
            if face is None:
                continue
            face_np = face.squeeze(0).permute(1, 2, 0).cpu().numpy()
            face_np = (np.clip(face_np, 0, 1) * 255).astype('uint8')
            out_path = os.path.join(out_person_dir, img_file)
            cv2.imwrite(out_path, cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR))
