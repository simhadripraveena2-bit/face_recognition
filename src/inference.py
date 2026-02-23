import os

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image

from src.face_pipeline import DEVICE as EMB_DEVICE, get_aligned_face_tensor
from src.feature_extraction import extract_geometric_analysis_from_image
from src.models import FaceClassifier, HybridFusionClassifier

MODEL_CHECKPOINT = os.getenv("MPD_CLASSIFIER_PATH", "models/classifier_checkpoint.pth")

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(EMB_DEVICE)

_classifier = None
label2name = {}
_embedding_prototypes = None
_geometry_prototypes = None
_scaler_mean = None
_scaler_scale = None
_emb_dim = 512
_geo_dim = 12


def load_classifier(checkpoint_path: str = MODEL_CHECKPOINT, num_classes: int = None, input_dim: int = None):
    global _classifier, label2name, _embedding_prototypes, _geometry_prototypes, _scaler_mean, _scaler_scale, _emb_dim, _geo_dim
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Classifier checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=EMB_DEVICE)
    model_type = ckpt.get("model_type", "mlp")
    _emb_dim = int(ckpt.get("emb_dim", 512))
    _geo_dim = int(ckpt.get("geo_dim", 12))

    if model_type == "hybrid_fusion":
        _classifier = HybridFusionClassifier(
            emb_dim=_emb_dim,
            geo_dim=_geo_dim,
            hidden_dim=ckpt.get("hidden_dim", 256),
            num_classes=ckpt.get('num_classes', num_classes),
        ).to(EMB_DEVICE)
    else:
        _classifier = FaceClassifier(
            input_dim=ckpt.get('input_dim', input_dim or (_emb_dim + _geo_dim)),
            hidden_dim=ckpt.get('hidden_dim', 256),
            num_classes=ckpt.get('num_classes', num_classes),
        ).to(EMB_DEVICE)

    _classifier.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    _classifier.eval()

    label2name = ckpt.get('label2name', ckpt.get('idx2label', {}))
    _embedding_prototypes = ckpt.get('embedding_prototypes')
    _geometry_prototypes = ckpt.get('geometry_prototypes')
    _scaler_mean = ckpt.get('scaler_mean')
    _scaler_scale = ckpt.get('scaler_scale')


def image_to_embedding(image: Image.Image):
    face_tensor = get_aligned_face_tensor(image)
    if face_tensor is None:
        return None
    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy().squeeze().astype(np.float32)
    return emb


def image_to_geometric_features(image: Image.Image):
    return extract_geometric_analysis_from_image(image)["vector"]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _apply_scaler(x: np.ndarray) -> np.ndarray:
    if _scaler_mean is None or _scaler_scale is None:
        return x
    mean = np.asarray(_scaler_mean, dtype=np.float32)
    scale = np.asarray(_scaler_scale, dtype=np.float32)
    return (x - mean) / np.maximum(scale, 1e-8)


def predict_from_image_pil(image: Image.Image, top_k: int = 3, enable_hybrid: bool = True) -> dict:
    global _classifier, label2name
    if _classifier is None:
        load_classifier()

    emb = image_to_embedding(image)
    if emb is None:
        return {"error": "No face found in the image."}

    geometry_analysis = extract_geometric_analysis_from_image(image)
    geo = geometry_analysis["vector"]

    combined_raw = np.concatenate([emb, geo], axis=0).astype(np.float32)
    combined = _apply_scaler(combined_raw)

    x = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(EMB_DEVICE)
    with torch.no_grad():
        logits = _classifier(x)
        cls_probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

    emb_sims, geo_sims = None, None
    if _embedding_prototypes is not None and _geometry_prototypes is not None:
        emb_part = combined[:_emb_dim]
        geo_part = combined[_emb_dim:_emb_dim + _geo_dim]
        emb_proto = np.asarray(_embedding_prototypes, dtype=np.float32)
        geo_proto = np.asarray(_geometry_prototypes, dtype=np.float32)
        emb_sims = np.array([_cosine_similarity(emb_part, p) for p in emb_proto], dtype=np.float32)
        geo_sims = np.array([_cosine_similarity(geo_part, p) for p in geo_proto], dtype=np.float32)

    if enable_hybrid:
        final_scores = cls_probs
    else:
        if emb_sims is not None:
            final_scores = (emb_sims + 1.0) / 2.0
        else:
            final_scores = cls_probs

    idx_sorted = np.argsort(final_scores)[::-1][:top_k]
    results = []
    for idx in idx_sorted:
        name = label2name.get(str(idx), label2name.get(idx, f"class_{idx}"))
        results.append({
            "name": name,
            "score": float(final_scores[idx]),
            "class_prob": float(cls_probs[idx]),
            "class_index": int(idx),
            "embedding_similarity": float(emb_sims[idx]) if emb_sims is not None else None,
            "geometry_similarity": float(geo_sims[idx]) if geo_sims is not None else None,
        })

    return {
        "predictions": results,
        "embedding": emb.tolist(),
        "embedding_dim": int(emb.shape[0]),
        "geometry_features": geo.tolist(),
        "geometry_distances": geometry_analysis["distances"],
        "landmarks": geometry_analysis["landmarks"],
        "hybrid_enabled": enable_hybrid,
    }
