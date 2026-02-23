import os
from typing import Optional

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

from src.feature_extraction import extract_geometric_analysis_from_image

MODEL_CHECKPOINT = os.getenv("MPD_CLASSIFIER_PATH", "models/classifier_checkpoint.pth")
EMB_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=EMB_DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(EMB_DEVICE)

_classifier = None
label2name = {}
_embedding_prototypes = None
_geometry_prototypes = None


def load_classifier(checkpoint_path: str = MODEL_CHECKPOINT, num_classes: int = None, input_dim: int = 517):
    global _classifier, label2name, _embedding_prototypes, _geometry_prototypes
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Classifier checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=EMB_DEVICE)
    from src.models import FaceClassifier

    nc = ckpt.get('num_classes', num_classes)
    _classifier = FaceClassifier(
        input_dim=ckpt.get('input_dim', input_dim),
        hidden_dim=ckpt.get('hidden_dim', 256),
        num_classes=nc,
    ).to(EMB_DEVICE)
    _classifier.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    _classifier.eval()

    label2name = ckpt.get('label2name', ckpt.get('idx2label', {}))
    _embedding_prototypes = ckpt.get('embedding_prototypes')
    _geometry_prototypes = ckpt.get('geometry_prototypes')


def image_to_embedding(image: Image.Image) -> Optional[np.ndarray]:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    face_tensor = mtcnn(np.array(image))
    if face_tensor is None:
        return None
    face_tensor = face_tensor.unsqueeze(0).to(EMB_DEVICE)
    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy().squeeze().astype(np.float32)
    return emb


def image_to_geometric_features(image: Image.Image) -> Optional[np.ndarray]:
    return extract_geometric_analysis_from_image(image)["vector"]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def predict_from_image_pil(image: Image.Image, top_k: int = 3, embedding_weight: float = 0.7, geometry_weight: float = 0.3, enable_hybrid: bool = True) -> dict:
    global _classifier, label2name
    if _classifier is None:
        load_classifier()

    emb = image_to_embedding(image)
    if emb is None:
        return {"error": "No face found in the image."}

    geometry_analysis = extract_geometric_analysis_from_image(image)
    geo = geometry_analysis["vector"]

    combined = np.concatenate([emb, geo], axis=0)
    x = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(EMB_DEVICE)

    with torch.no_grad():
        logits = _classifier(x)
        cls_probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

    weighted_scores = cls_probs.copy()
    emb_sims = None
    geo_sims = None

    if enable_hybrid and _embedding_prototypes is not None and _geometry_prototypes is not None:
        emb_proto = np.asarray(_embedding_prototypes, dtype=np.float32)
        geo_proto = np.asarray(_geometry_prototypes, dtype=np.float32)
        emb_sims = np.array([_cosine_similarity(emb, p) for p in emb_proto], dtype=np.float32)
        geo_sims = np.array([_cosine_similarity(geo, p) for p in geo_proto], dtype=np.float32)

        emb_norm = (emb_sims + 1.0) / 2.0
        geo_norm = (geo_sims + 1.0) / 2.0
        weighted_scores = embedding_weight * emb_norm + geometry_weight * geo_norm

    idx_sorted = np.argsort(weighted_scores)[::-1][:top_k]
    results = []
    for idx in idx_sorted:
        name = label2name.get(str(idx), label2name.get(idx, f"class_{idx}"))
        item = {
            "name": name,
            "score": float(weighted_scores[idx]),
            "class_prob": float(cls_probs[idx]),
            "class_index": int(idx),
            "embedding_similarity": float(emb_sims[idx]) if emb_sims is not None else None,
            "geometry_similarity": float(geo_sims[idx]) if geo_sims is not None else None,
        }
        results.append(item)

    return {
        "predictions": results,
        "embedding": emb.tolist(),
        "embedding_dim": int(emb.shape[0]),
        "geometry_features": geo.tolist(),
        "geometry_distances": geometry_analysis["distances"],
        "landmarks": geometry_analysis["landmarks"],
        "hybrid_enabled": enable_hybrid,
    }
