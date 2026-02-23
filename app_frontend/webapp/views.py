# app_frontend/webapp/views.py
import requests
from django.shortcuts import render, redirect

BACKEND_EMBED_URL = "http://127.0.0.1:8000/embed/"  # change if backend hosted elsewhere


def index(request):
    return render(request, "webapp/index.html")


def upload(request):
    if request.method == "POST" and request.FILES.get("image"):
        f = request.FILES["image"]
        files = {'file': (f.name, f.read(), f.content_type)}
        try:
            resp = requests.post(BACKEND_EMBED_URL, files=files, timeout=20)
            result = resp.json()
        except Exception as exc:
            result = {"error": str(exc)}
        return render(request, "webapp/index.html", {"result": result})
    return redirect("index")
