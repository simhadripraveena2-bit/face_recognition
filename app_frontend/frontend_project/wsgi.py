# app_frontend/frontend_project/wsgi.py
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'frontend_project.settings')
application = get_wsgi_application()
