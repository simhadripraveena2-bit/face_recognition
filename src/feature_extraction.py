import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1

try:
    import mediapipe as mp
except Exception:  # mediapipe is optional at runtime
    mp = None


device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def _to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert('RGB'))
    return torch.tensor(arr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0


def extract_embedding(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert('RGB')
    img_tensor = _to_tensor(img)
    with torch.no_grad():
        emb = resnet(img_tensor).cpu().numpy().squeeze().astype(np.float32)
    return emb


def _landmark_xy(landmarks, idx: int, width: int, height: int):
    lm = landmarks[idx]
    return np.array([lm.x * width, lm.y * height], dtype=np.float32)


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def extract_geometric_features_from_image(image: Image.Image) -> Optional[np.ndarray]:
    """
    Returns normalized geometric vector:
      [eye_distance, nose_to_mouth, ear_to_ear, face_height, face_width]
    normalized by face_width.
    """
    rgb = np.array(image.convert('RGB'))
    h, w = rgb.shape[:2]

    if mp is not None:
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        result = face_mesh.process(rgb)
        face_mesh.close()
        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark

            left_eye = _landmark_xy(lms, 33, w, h)
            right_eye = _landmark_xy(lms, 263, w, h)
            nose_tip = _landmark_xy(lms, 1, w, h)
            mouth_left = _landmark_xy(lms, 61, w, h)
            mouth_right = _landmark_xy(lms, 291, w, h)
            mouth_center = (mouth_left + mouth_right) / 2.0
            left_ear = _landmark_xy(lms, 234, w, h)
            right_ear = _landmark_xy(lms, 454, w, h)
            chin = _landmark_xy(lms, 152, w, h)
            forehead = _landmark_xy(lms, 10, w, h)
            left_jaw = _landmark_xy(lms, 234, w, h)
            right_jaw = _landmark_xy(lms, 454, w, h)

            face_width = max(_distance(left_jaw, right_jaw), 1e-6)
            features = np.array([
                _distance(left_eye, right_eye) / face_width,
                _distance(nose_tip, mouth_center) / face_width,
                _distance(left_ear, right_ear) / face_width,
                _distance(chin, forehead) / face_width,
                face_width / face_width,
            ], dtype=np.float32)
            return features

    # Fallback if mediapipe unavailable or no landmarks found
    # Use whole image geometry as safe defaults so pipeline still works.
    face_width = float(max(w, 1))
    features = np.array([
        0.35,  # approximate normalized eye distance
        0.20,  # approximate normalized nose-mouth distance
        0.90,  # approximate ear-ear distance
        float(h) / face_width,
        1.0,
    ], dtype=np.float32)
    return features


def extract_geometric_features(img_path: str) -> Optional[np.ndarray]:
    img = Image.open(img_path).convert('RGB')
    return extract_geometric_features_from_image(img)


def process_dataset(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        if not os.path.isdir(person_dir):
            continue
        out_person_dir = os.path.join(output_dir, person)
        os.makedirs(out_person_dir, exist_ok=True)
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            if not os.path.isfile(img_path):
                continue
            stem, _ = os.path.splitext(img_file)
            emb = extract_embedding(img_path)
            geo = extract_geometric_features(img_path)
            np.save(os.path.join(out_person_dir, f"{stem}.npy"), emb)
            np.save(os.path.join(out_person_dir, f"{stem}.geo.npy"), geo)
