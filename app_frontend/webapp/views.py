# app_frontend/webapp/views.py
import requests
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse

BACKEND_PREDICT_URL = "http://127.0.0.1:8000/predict"  # change if backend hosted elsewhere

def index(request):
    return render(request, "webapp/index.html")

def upload(request):
    if request.method == "POST" and request.FILES.get("image"):
        f = request.FILES["image"]
        files = {'file': (f.name, f.read(), f.content_type)}
        try:
            resp = requests.post(BACKEND_PREDICT_URL, files=files, timeout=20)
            result = resp.json()
        except Exception as e:
            result = {"error": str(e)}
        return render(request, "webapp/index.html", {"result": result})
    return redirect("index")
