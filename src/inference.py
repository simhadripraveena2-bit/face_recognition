# src/inference.py
import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from typing import Tuple

# Paths (relative to repo root)
MODEL_CHECKPOINT = os.getenv("MPD_CLASSIFIER_PATH", "models/classifier_checkpoint.pth")
EMB_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize detector & embedding model
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=EMB_DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(EMB_DEVICE)

# Classifier will be loaded lazily
_classifier = None
label2name = {}  # filled from dataset folder structure or from checkpoint

def load_classifier(checkpoint_path: str = MODEL_CHECKPOINT, num_classes: int = None, input_dim: int = 512):
    """
    Loads classifier checkpoint. The checkpoint should contain:
     - 'state_dict' : model state dict
     - 'label2name' : mapping idx -> person_name (optional)
    If checkpoint doesn't have label info, infer from data folder if possible.
    """
    global _classifier, label2name
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Classifier checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=EMB_DEVICE)
    # dynamic import of your FaceClassifier if using the same name
    from src.models import FaceClassifier
    if 'num_classes' in ckpt:
        nc = ckpt['num_classes']
    else:
        nc = num_classes if num_classes is not None else (ckpt.get('state_dict')['net.0.weight'].shape[0] if 'state_dict' in ckpt else None)

    _classifier = FaceClassifier(input_dim=input_dim, hidden_dim=ckpt.get('hidden_dim', 256), num_classes=nc).to(EMB_DEVICE)
    if 'state_dict' in ckpt:
        _classifier.load_state_dict(ckpt['state_dict'])
    else:
        # If full model saved
        try:
            _classifier.load_state_dict(ckpt)
        except Exception:
            raise RuntimeError("Checkpoint format not recognized; expected dict with 'state_dict' or raw state_dict")
    _classifier.eval()

    # load label mapping if provided
    if 'label2name' in ckpt:
        label2name = ckpt['label2name']
    elif 'idx2label' in ckpt:
        label2name = ckpt['idx2label']
    # else will be filled by external code or remain empty

def image_to_embedding(image: Image.Image) -> np.ndarray:
    """
    Given a PIL image, detect largest face, return 512-d embedding (numpy).
    """
    # Convert to RGB PIL if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # face detection via mtcnn returns a torch tensor [3,160,160] or None
    face_tensor = mtcnn(np.array(image))
    if face_tensor is None:
        return None
    face_tensor = face_tensor.unsqueeze(0).to(EMB_DEVICE)
    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy().squeeze()  # shape (512,)
    return emb

def predict_from_image_pil(image: Image.Image, top_k: int = 3) -> dict:
    """
    Run full pipeline: detect face -> embedding -> classifier -> return top_k predictions (name, score)
    """
    global _classifier, label2name
    if _classifier is None:
        load_classifier()

    emb = image_to_embedding(image)
    if emb is None:
        return {"error": "No face found in the image."}

    x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(EMB_DEVICE)
    with torch.no_grad():
        logits = _classifier(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()  # (num_classes,)

    # get top_k
    idx_sorted = np.argsort(probs)[::-1][:top_k]
    results = []
    for idx in idx_sorted:
        name = label2name.get(str(idx), label2name.get(idx, f"class_{idx}"))
        results.append({"name": name, "score": float(probs[idx]), "class_index": int(idx)})
    return {"predictions": results}
