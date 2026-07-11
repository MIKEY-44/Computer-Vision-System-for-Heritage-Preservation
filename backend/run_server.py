#!/usr/bin/env python
"""
Pushkarani Backend Launcher - With Complete Error Handling
"""
import sys
import os

# Ensure we're in the right directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(backend_dir)

print("="*70)
print("Pushkarani Backend - Starting")
print("="*70)
print(f"Working directory: {os.getcwd()}")
print(f"Python: {sys.executable}")
print()

try:
    print("[STEP 1] Importing Flask app...")
    from app import app
    print("✓ Flask app imported")
    
    print("\n[STEP 2] Starting Waitress server on 127.0.0.1:5000...")
    from waitress import serve
    print("✓ Waitress imported")
    
    print("\n[STEP 3] Server starting (press Ctrl+C to stop)...")
    print("="*70)
    serve(app, host='127.0.0.1', port=5000, _quiet=True)
    
except ImportError as e:
    print(f"\n✗ Import Error: {e}")
    print("\nMissing dependency. Install with:")
    if 'waitress' in str(e):
        print("  pip install waitress")
    elif 'flask' in str(e):
        print("  pip install flask flask-cors flask-caching python-dotenv")
    else:
        print(f"  pip install {str(e).split()[-1]}")
    sys.exit(1)
    
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
