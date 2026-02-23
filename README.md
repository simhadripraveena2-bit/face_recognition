# Face Recognition (Embedding + Geometry Hybrid)

An end-to-end face recognition project with:
- a **training pipeline** (crop faces → extract features → train classifier),
- a **FastAPI backend** for `/embed/` and `/predict/`, and
- a **Django frontend** for uploading images and viewing recognition results.

The system combines:
- **512-D deep face embeddings** (FaceNet/InceptionResnetV1), and
- **geometric facial features** (MediaPipe landmarks-derived ratios/distances)

for hybrid identity prediction.

---

## Project structure

```text
face_recognition/
├── app.py                          # Compatibility entrypoint -> app_backend.main:app
├── train.py                        # Compatibility entrypoint -> src.training.train_model
├── model.py                        # Compatibility entrypoint -> src.models
├── README.md
├── requirements.txt
│
├── models/
│   └── classifier_checkpoint.pth   # Saved hybrid classifier checkpoint
│
├── src/
│   ├── main.py                     # Local training/evaluation pipeline script
│   ├── preprocessing.py            # Face detection + crop using MTCNN
│   ├── feature_extraction.py       # Embeddings + geometric features extraction
│   ├── face_pipeline.py            # Shared aligned-face utility
│   ├── datasets.py                 # Dataset loader for .npy embeddings
│   ├── models.py                   # FaceClassifier + HybridFusionClassifier
│   ├── training.py                 # Hybrid training + checkpoint/prototype save
│   ├── evaluation.py               # Accuracy evaluation helper
│   ├── inference.py                # Runtime inference + top-k predictions
│   └── utils.py
│
├── app_backend/
│   ├── main.py                     # FastAPI app
│   ├── requirements.txt
│   └── face_features.db            # Auto-created SQLite table for extracted features
│
├── app_frontend/
│   ├── manage.py
│   ├── requirements.txt
│   ├── frontend_project/
│   └── webapp/
│       ├── views.py                # Proxies upload to backend /embed or /predict
│       ├── urls.py
│       └── templates/webapp/index.html
│
├── scripts/
│   ├── evaluate_models.py          # Embedding vs geometry vs hybrid metrics
│   ├── evaluate_benchmarks.py      # Simple rank-1 benchmark helper
│   ├── visualize_features.py       # Landmark overlay + feature charts
│   └── ablation_study.py           # Prototype shape/ablation summary
│
└── docs/
    ├── research.md
    └── hybrid_fusion_examples.md
```

---

## How the pipeline works

1. **Preprocessing** (`src/preprocessing.py`)
   - Uses MTCNN to detect/crop faces from images organized by person folders.

2. **Feature extraction** (`src/feature_extraction.py`)
   - Extracts a 512-dimensional embedding with `InceptionResnetV1(pretrained='vggface2')`.
   - Extracts geometric features (landmark-based normalized distances/ratios) using MediaPipe FaceMesh.
   - Saves per-image features as:
     - `<sample>.npy` (embedding)
     - `<sample>.geo.npy` (geometry)

3. **Dataset + training** (`src/datasets.py`, `src/training.py`)
   - Loads embedding and geometry vectors.
   - Trains `HybridFusionClassifier` (embedding branch + geometry branch + gating fusion head).
   - Saves best checkpoint to `models/classifier_checkpoint.pth`.

4. **Inference API** (`app_backend/main.py`, `src/inference.py`)
   - `/embed/`: returns embedding + geometry and stores feature rows in SQLite.
   - `/predict/`: returns top-k predictions, scores, similarities, and detected landmarks.

5. **Frontend UI** (`app_frontend/webapp/templates/webapp/index.html`)
   - Upload image from browser.
   - Toggle hybrid mode.
   - View top predictions, geometric distances, and landmark overlay.

---

## Data layout expected

Your raw dataset should be identity-folder style:

```text
data/
└── vggface2/
    ├── person_1/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── person_2/
        └── img3.jpg
```

After preprocessing + extraction, features are expected in:

```text
data/embeddings/<person>/<sample>.npy
data/embeddings/<person>/<sample>.geo.npy
```

---

## Setup

### 1) Install base dependencies

```bash
pip install -r requirements.txt
```

### 2) (Optional) Install backend/frontend specific dependencies

```bash
pip install -r app_backend/requirements.txt
pip install -r app_frontend/requirements.txt
```

---

## Training / offline pipeline

Run the main script:

```bash
python src/main.py
```

This script is configured to:
- crop faces from `data/vggface2` to `data/faces_cropped`,
- extract embeddings/features into `data/embeddings`,
- train a classifier,
- print evaluation metrics.

> Note: It expects the dataset folders to exist locally.

---

## Run backend API

```bash
uvicorn app_backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

- `POST /embed/`
  - Input: image file
  - Output: embedding vector, embedding dimension, geometry features

- `POST /predict/?top_k=3&enable_hybrid=true`
  - Input: image file
  - Output: top-k predictions with score details, embedding, geometry distances, and landmarks

---

## Run frontend

```bash
cd app_frontend
python manage.py runserver 8001
```

Open: `http://127.0.0.1:8001/`

Make sure backend is already running on `http://127.0.0.1:8000/`.

---

## Evaluation & analysis scripts

### Compare embedding vs geometry vs hybrid

```bash
python scripts/evaluate_models.py --emb_dir data/embeddings
```

Generates:
- `artifacts/evaluation_results.csv`
- `artifacts/evaluation_comparison.png`

### Visualize landmarks and geometry feature magnitudes

```bash
python scripts/visualize_features.py --image path/to/face.jpg \
  --overlay_out artifacts/landmark_overlay.png \
  --importance_out artifacts/feature_importance.png
```

### Quick prototype ablation summary from checkpoint

```bash
python scripts/ablation_study.py --checkpoint models/classifier_checkpoint.pth
```

---

## Notes

- The backend creates/uses SQLite DB at `app_backend/face_features.db` to log extracted features.
- You can use `app.py` as a compatibility ASGI entrypoint (`from app import app`).
- `train.py` and `model.py` are compatibility modules that re-export training/model objects from `src/`.

