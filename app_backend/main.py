import io
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from src.feature_extraction import extract_embedding
from src.utils import identify_person

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes and decode
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Convert to RGB for consistency with PIL-based model
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Save temporarily (so we can reuse extract_embedding)
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, img_rgb)

        # Extract embedding using pretrained model
        emb = extract_embedding(temp_path)

        # Identify the person
        result = identify_person(emb)

        return {
            "name": result.get("name"),
            "confidence": result.get("confidence")
        }

    except Exception as e:
        return {"detail": str(e)}
