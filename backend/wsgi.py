"""
WSGI entry point for Gunicorn
Run with: gunicorn -w 1 -b 0.0.0.0:5000 wsgi:app
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Flask app
from app import app

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
