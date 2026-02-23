import os

import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from src.face_pipeline import DEVICE, get_aligned_face_tensor

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)


def extract_embedding_from_image(image: Image.Image):
    face_tensor = get_aligned_face_tensor(image)
    if face_tensor is None:
        return None
    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy().squeeze().astype(np.float32)
    return emb


def extract_embedding(img_path: str):
    return extract_embedding_from_image(Image.open(img_path).convert('RGB'))


def _xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _ratio(a, b):
    return float(a / max(b, 1e-6))


def extract_geometric_analysis_from_image(image: Image.Image) -> dict:
    rgb = np.array(image.convert('RGB'))
    h, w = rgb.shape[:2]

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    result = face_mesh.process(rgb)
    face_mesh.close()

    if result.multi_face_landmarks:
        lms = result.multi_face_landmarks[0].landmark
        p = {
            "left_eye": _xy(lms, 33, w, h),
            "right_eye": _xy(lms, 263, w, h),
            "nose_tip": _xy(lms, 1, w, h),
            "nose_bridge": _xy(lms, 6, w, h),
            "mouth_left": _xy(lms, 61, w, h),
            "mouth_right": _xy(lms, 291, w, h),
            "left_jaw": _xy(lms, 234, w, h),
            "right_jaw": _xy(lms, 454, w, h),
            "left_cheek": _xy(lms, 93, w, h),
            "right_cheek": _xy(lms, 323, w, h),
            "chin": _xy(lms, 152, w, h),
            "forehead": _xy(lms, 10, w, h),
        }
        p["mouth_center"] = (p["mouth_left"] + p["mouth_right"]) / 2.0

        face_width = max(_dist(p["left_jaw"], p["right_jaw"]), 1e-6)
        eye_dist = _dist(p["left_eye"], p["right_eye"])
        nose_mouth = _dist(p["nose_tip"], p["mouth_center"])
        face_h = _dist(p["chin"], p["forehead"])
        left_eye_nose = _dist(p["left_eye"], p["nose_tip"])
        right_eye_nose = _dist(p["right_eye"], p["nose_tip"])
        left_mouth_nose = _dist(p["mouth_left"], p["nose_tip"])
        right_mouth_nose = _dist(p["mouth_right"], p["nose_tip"])
        cheek_width = _dist(p["left_cheek"], p["right_cheek"])

        nose_vec = p["nose_tip"] - p["nose_bridge"]
        nose_angle = float(np.arctan2(nose_vec[1], nose_vec[0]) / np.pi)  # normalized [-1, 1]

        distances = {
            "eye_distance": eye_dist / face_width,
            "nose_to_mouth": nose_mouth / face_width,
            "ear_to_ear": face_width / face_width,
            "face_height": face_h / face_width,
            "face_width": face_width / face_width,
            "interocular_ratio": _ratio(eye_dist, face_h),
            "nose_angle": nose_angle,
            "facial_symmetry_eye": _ratio(abs(left_eye_nose - right_eye_nose), face_width),
            "facial_symmetry_mouth": _ratio(abs(left_mouth_nose - right_mouth_nose), face_width),
            "eye_mouth_ratio": _ratio(eye_dist, max(_dist(p["mouth_left"], p["mouth_right"]), 1e-6)),
            "cheek_to_jaw_ratio": _ratio(cheek_width, face_width),
            "nose_mouth_to_height_ratio": _ratio(nose_mouth, face_h),
        }

        vector = np.array(list(distances.values()), dtype=np.float32)
        return {"landmarks": {k: v.tolist() for k, v in p.items()}, "distances": distances, "vector": vector}

    # deterministic fallback
    distances = {
        "eye_distance": 0.35,
        "nose_to_mouth": 0.20,
        "ear_to_ear": 1.0,
        "face_height": float(h) / max(float(w), 1.0),
        "face_width": 1.0,
        "interocular_ratio": 0.40,
        "nose_angle": 0.0,
        "facial_symmetry_eye": 0.0,
        "facial_symmetry_mouth": 0.0,
        "eye_mouth_ratio": 1.0,
        "cheek_to_jaw_ratio": 1.0,
        "nose_mouth_to_height_ratio": 0.25,
    }
    return {"landmarks": {}, "distances": distances, "vector": np.array(list(distances.values()), dtype=np.float32)}


def extract_geometric_features_from_image(image: Image.Image):
    return extract_geometric_analysis_from_image(image)["vector"]


def extract_geometric_features(img_path: str):
    return extract_geometric_features_from_image(Image.open(img_path).convert('RGB'))

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
            image = Image.open(img_path).convert('RGB')
            emb = extract_embedding_from_image(image)
            if emb is None:
                continue
            geo = extract_geometric_features_from_image(image)
            np.save(os.path.join(out_person_dir, f"{stem}.npy"), emb)
            np.save(os.path.join(out_person_dir, f"{stem}.geo.npy"), geo)
