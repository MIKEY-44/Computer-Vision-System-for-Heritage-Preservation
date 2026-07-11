#!/usr/bin/env python3
"""
Pushkarani Backend - Robust Windows-Compatible Server Launcher
Attempts to start backend with multiple fallback strategies
"""

import sys
import os
import traceback
from pathlib import Path

def setup_path():
    """Add backend to path"""
    backend_dir = Path(__file__).parent
    if backend_dir not in sys.path:
        sys.path.insert(0, str(backend_dir))

def attempt_import_flask():
    """Attempt to import Flask"""
    try:
        import flask
        print("[OK] Flask imported successfully")
        return True
    except ImportError as e:
        print(f"[ERROR] Flask import failed: {e}")
        return False

def attempt_import_models():
    """Verify model files exist"""
    model_paths = [
        'densenet/final_model.keras',
        'inception/final_model.keras',
        'mobilenet/final_model.keras',
        'stone_quality_model/final_model.keras'
    ]
    
    models_dir = Path(__file__).parent.parent
    
    print("\nChecking model files:")
    all_exist = True
    for model_path in model_paths:
        full_path = models_dir / model_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {model_path} (NOT FOUND)")
            all_exist = False
    
    return all_exist

def run_with_flask_dev():
    """Strategy 1: Run with Flask development server"""
    print("\n[STRATEGY 1] Attempting Flask development server...")
    try:
        from app import app
        print("✓ App imported successfully")
        
        print("✓ Starting Flask development server on http://127.0.0.1:5000")
        print("✓ Press Ctrl+C to stop\n")
        
        # Run with minimal threading
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=False,
            processes=1
        )
        return True
    except Exception as e:
        print(f"✗ Flask dev server failed: {e}")
        traceback.print_exc()
        return False

def run_with_waitress():
    """Strategy 2: Run with Waitress WSGI server"""
    print("\n[STRATEGY 2] Attempting Waitress WSGI server...")
    try:
        from waitress import serve
        from app import app
        print("✓ Waitress and app imported successfully")
        
        print("✓ Starting Waitress server on http://127.0.0.1:5000")
        print("✓ Press Ctrl+C to stop\n")
        
        # Waitress configuration optimized for Windows
        serve(
            app,
            host='127.0.0.1',
            port=5000,
            threads=2,
            _quiet=False
        )
        return True
    except Exception as e:
        print(f"✗ Waitress server failed: {e}")
        traceback.print_exc()
        return False

def run_with_gunicorn():
    """Strategy 3: Run with Gunicorn (if available)"""
    print("\n[STRATEGY 3] Attempting Gunicorn server...")
    try:
        import subprocess
        print("✓ Starting Gunicorn server on http://127.0.0.1:5000")
        print("✓ Press Ctrl+C to stop\n")
        
        result = subprocess.run([
            sys.executable, '-m', 'gunicorn',
            '--bind', '127.0.0.1:5000',
            '--workers', '2',
            '--threads', '2',
            '--worker-class', 'gthread',
            '--timeout', '120',
            'app:app'
        ], check=False)
        
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Gunicorn server failed: {e}")
        return False

def main():
    print("=" * 70)
    print("PUSHKARANI BACKEND SERVER LAUNCHER")
    print("=" * 70)
    
    # Setup
    setup_path()
    
    # Pre-flight checks
    print("\n[PREFLIGHT CHECKS]")
    
    if not attempt_import_flask():
        print("\n✗ CRITICAL: Flask not installed")
        print("  Run: pip install -r requirements.txt")
        sys.exit(1)
    
    if not attempt_import_models():
        print("\n⚠ WARNING: Some model files missing")
        print("  Ensure model directories exist in parent directory")
    
    # Try deployment strategies in order
    print("\n[DEPLOYMENT STRATEGIES]")
    print("Attempting to start backend service...\n")
    
    strategies = [
        ("Waitress WSGI Server (Recommended)", run_with_waitress),
        ("Flask Development Server (Fallback)", run_with_flask_dev),
        ("Gunicorn (Linux-style)", run_with_gunicorn),
    ]
    
    for strategy_name, strategy_func in strategies:
        try:
            print(f"\n{'=' * 70}")
            if strategy_func():
                print("✓ Server running successfully")
                break
        except KeyboardInterrupt:
            print("\n✓ Server stopped by user")
            sys.exit(0)
        except Exception as e:
            print(f"✗ Strategy failed: {e}")
            continue
    else:
        print("\n" + "=" * 70)
        print("✗ All strategies failed")
        print("\nTROUBLESHOOTING:")
        print("  1. Verify all dependencies: pip install -r requirements.txt")
        print("  2. Check model files exist in ../convnext, ../densenet, etc.")
        print("  3. Try running with Docker: docker-compose up -d")
        print("  4. Try WSL2 instead of native Windows")
        sys.exit(1)

if __name__ == '__main__':
    main()
