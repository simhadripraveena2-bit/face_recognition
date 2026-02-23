# app_frontend/webapp/views.py
import base64
import json

import requests
from django.shortcuts import render, redirect

BACKEND_EMBED_URL = "http://127.0.0.1:8000/embed/"
BACKEND_PREDICT_URL = "http://127.0.0.1:8000/predict/"


def index(request):
    return render(request, "webapp/index.html")


def upload(request):
    if request.method == "POST" and request.FILES.get("image"):
        f = request.FILES["image"]
        raw = f.read()
        files = {'file': (f.name, raw, f.content_type)}
        enable_hybrid = request.POST.get("enable_hybrid") == "on"
        try:
            if enable_hybrid:
                resp = requests.post(
                    BACKEND_PREDICT_URL,
                    params={"enable_hybrid": "true", "top_k": 3},
                    files=files,
                    timeout=20,
                )
            else:
                resp = requests.post(BACKEND_EMBED_URL, files=files, timeout=20)
            result = resp.json()
        except Exception as exc:
            result = {"error": str(exc)}

        image_data_url = f"data:{f.content_type or 'image/jpeg'};base64,{base64.b64encode(raw).decode('utf-8')}"
        return render(
            request,
            "webapp/index.html",
            {
                "result": result,
                "image_data_url": image_data_url,
                "enable_hybrid": enable_hybrid,
                "landmarks_json": json.dumps(result.get("landmarks", {})),
            },
        )
    return redirect("index")
