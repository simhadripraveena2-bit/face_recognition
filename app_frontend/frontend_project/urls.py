# app_frontend/frontend_project/urls.py
from django.urls import path, include

urlpatterns = [
    path('', include('webapp.urls')),
]
