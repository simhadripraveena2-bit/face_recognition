import json
import sqlite3
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from src.inference import (
    image_to_embedding,
    image_to_geometric_features,
    predict_from_image_pil,
)

app = FastAPI(title="Face Embedding API")
DB_PATH = "app_backend/face_features.db"


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS face_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            geometry_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _save_feature_row(embedding, geometry):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO face_features (created_at, embedding_json, geometry_json) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(), json.dumps(embedding), json.dumps(geometry)),
    )
    conn.commit()
    conn.close()


def _read_image_from_upload(file: UploadFile) -> Image.Image:
    contents = file.file.read()
    if not contents:
        raise ValueError("Uploaded file is empty.")
    return Image.open(BytesIO(contents)).convert("RGB")


_init_db()


@app.post("/embed/")
async def embed(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        embedding = image_to_embedding(image)
        if embedding is None:
            return {"error": "No face found in the image."}

        geometry = image_to_geometric_features(image)
        if geometry is None:
            geometry = [0.0] * 5
        else:
            geometry = geometry.tolist()

        vector = embedding.tolist()
        _save_feature_row(vector, geometry)
        return {
            "embedding": vector,
            "embedding_dim": len(vector),
            "geometry_features": geometry,
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/predict/")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    try:
        image = _read_image_from_upload(file)
        result = predict_from_image_pil(image, top_k=top_k, embedding_weight=0.7, geometry_weight=0.3)
        if "error" in result:
            return result

        _save_feature_row(result.get("embedding", []), result.get("geometry_features", []))
        return result
    except Exception as exc:
        return {"error": str(exc)}
