from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from src.inference import image_to_embedding, predict_from_image_pil

app = FastAPI(title="Face Embedding API")


def _read_image_from_upload(file: UploadFile) -> Image.Image:
    contents = file.file.read()
    if not contents:
        raise ValueError("Uploaded file is empty.")
    image = Image.open(BytesIO(contents)).convert("RGB")
    return image


@app.post("/embed/")
async def embed(file: UploadFile = File(...)):
    try:
        image = _read_image_from_upload(file)
        embedding = image_to_embedding(image)
        if embedding is None:
            return {"error": "No face found in the image."}

        vector = embedding.tolist()
        return {
            "embedding": vector,
            "embedding_dim": len(vector),
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/predict/")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    try:
        image = _read_image_from_upload(file)
        result = predict_from_image_pil(image, top_k=top_k)
        if "error" in result:
            return result

        embedding = image_to_embedding(image)
        result["embedding"] = embedding.tolist() if embedding is not None else None
        result["embedding_dim"] = len(result["embedding"]) if result["embedding"] else 0
        return result
    except Exception as exc:
        return {"error": str(exc)}
