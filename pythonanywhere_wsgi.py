"""
Kisan Ki Awaz - PythonAnywhere WSGI entry point.

PythonAnywhere free tier runs WSGI applications. This file wraps the
FastAPI ASGI app using a2wsgi so it can be served on PythonAnywhere.

In your PythonAnywhere Web tab:
- Source code: /home/YOUR_USERNAME/kisan-ki-awaz
- Working directory: /home/YOUR_USERNAME/kisan-ki-awaz
- WSGI configuration file: /var/www/YOUR_USERNAME_pythonanywhere_com_wsgi.py
  (edit it to import from pythonanywhere_wsgi)
"""
import os
import sys

# PythonAnywhere sets the working directory automatically, but add it just in case.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from a2wsgi import ASGIMiddleware
from api import app

# This is the WSGI application object PythonAnywhere will look for.
application = ASGIMiddleware(app)
