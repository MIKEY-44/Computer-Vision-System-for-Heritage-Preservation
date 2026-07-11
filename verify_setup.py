"""
Pushkarani System Verification Script
Checks all requirements and configurations
"""

import os
import sys
import subprocess
import json
from pathlib import Path

class VerificationChecker:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success = []
        self.root_path = Path(__file__).parent
        
    def check_python(self):
        """Check Python version and packages"""
        print("\n[1/7] Checking Python environment...")
        try:
            import tensorflow as tf
            print(f"  ✓ TensorFlow {tf.__version__}")
            self.success.append("TensorFlow installed")
        except ImportError:
            self.errors.append("TensorFlow not installed - run: pip install tensorflow")
        
        try:
            import flask
            print(f"  ✓ Flask {flask.__version__}")
            self.success.append("Flask installed")
        except ImportError:
            self.errors.append("Flask not installed - run: pip install flask")
        
        try:
            import numpy
            print(f"  ✓ NumPy {numpy.__version__}")
        except ImportError:
            self.errors.append("NumPy not installed")
    
    def check_models(self):
        """Check if trained models exist"""
        print("\n[2/7] Checking trained models...")
        model_dirs = [
            'densenet', 'efficientnetv2', 'convnext', 'vgg16',
            'resnet50', 'mobilenet', 'mobilenetv3', 'inception',
            'swin', 'dinov2'
        ]
        
        found_models = 0
        for model_dir in model_dirs:
            model_path = self.root_path / model_dir / 'best_model.keras'
            if model_path.exists():
                print(f"  ✓ {model_dir}")
                found_models += 1
            else:
                self.warnings.append(f"Model not found: {model_dir}/best_model.keras")
        
        print(f"  Found {found_models}/{len(model_dirs)} models")
        if found_models == 0:
            self.errors.append("No trained models found - run training scripts first")
    
    def check_dataset(self):
        """Check dataset structure"""
        print("\n[3/7] Checking dataset...")
        dataset_path = self.root_path / 'dataset'
        
        if not dataset_path.exists():
            self.warnings.append("Dataset folder not found")
            return
        
        classes = ['type-1', 'type-2', 'type-3']
        for cls in classes:
            cls_path = dataset_path / cls
            if cls_path.exists():
                count = len(list(cls_path.glob('*.*')))
                print(f"  ✓ {cls}: {count} images")
            else:
                self.warnings.append(f"Dataset class not found: {cls}")
    
    def check_frontend(self):
        """Check frontend setup"""
        print("\n[4/7] Checking frontend...")
        frontend_path = self.root_path / 'frontend'
        
        if not (frontend_path / 'package.json').exists():
            self.errors.append("Frontend package.json not found")
            return
        
        print(f"  ✓ package.json found")
        
        if (frontend_path / 'node_modules').exists():
            print(f"  ✓ node_modules found")
            self.success.append("Frontend dependencies installed")
        else:
            self.warnings.append("Frontend dependencies not installed - run: cd frontend && npm install")
        
        if (frontend_path / '.env').exists():
            print(f"  ✓ .env configuration found")
    
    def check_backend(self):
        """Check backend setup"""
        print("\n[5/7] Checking backend...")
        backend_path = self.root_path / 'backend'
        
        if not (backend_path / 'app.py').exists():
            self.errors.append("Backend app.py not found")
            return
        
        print(f"  ✓ app.py found")
        
        if (backend_path / 'requirements.txt').exists():
            print(f"  ✓ requirements.txt found")
        
        venv_path = backend_path / 'venv'
        if venv_path.exists():
            print(f"  ✓ Virtual environment found")
            self.success.append("Backend virtual environment created")
        else:
            self.warnings.append("Virtual environment not found - run setup script")
    
    def check_ports(self):
        """Check if required ports are available"""
        print("\n[6/7] Checking ports...")
        ports = {5000: 'Backend', 3000: 'Frontend'}
        
        for port, service in ports.items():
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                
                if result == 0:
                    self.warnings.append(f"Port {port} ({service}) is already in use")
                else:
                    print(f"  ✓ Port {port} ({service}) available")
            except Exception as e:
                self.warnings.append(f"Could not check port {port}: {e}")
    
    def check_configuration(self):
        """Check configuration files"""
        print("\n[7/7] Checking configuration...")
        
        backend_env = self.root_path / 'backend' / '.env'
        frontend_env = self.root_path / 'frontend' / '.env'
        
        if backend_env.exists():
            print(f"  ✓ Backend .env found")
        else:
            self.warnings.append("Backend .env not found")
        
        if frontend_env.exists():
            print(f"  ✓ Frontend .env found")
        else:
            self.warnings.append("Frontend .env not found")
    
    def print_report(self):
        """Print verification report"""
        print("\n" + "="*60)
        print("VERIFICATION REPORT")
        print("="*60)
        
        if self.success:
            print(f"\n✓ SUCCESS ({len(self.success)}):")
            for msg in self.success:
                print(f"  • {msg}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"  • {msg}")
        
        if self.errors:
            print(f"\n✗ ERRORS ({len(self.errors)}):")
            for msg in self.errors:
                print(f"  • {msg}")
        
        print("\n" + "="*60)
        
        if self.errors:
            print("❌ SETUP INCOMPLETE - Fix errors above")
            return False
        elif self.warnings:
            print("⚠️  SETUP WITH WARNINGS - Some features may not work")
            return True
        else:
            print("✅ SETUP COMPLETE - Ready to run!")
            return True
    
    def run(self):
        """Run all checks"""
        print("╔════════════════════════════════════════════════════════╗")
        print("║  Pushkarani Classification System - Verification Tool  ║")
        print("╚════════════════════════════════════════════════════════╝")
        
        self.check_python()
        self.check_models()
        self.check_dataset()
        self.check_frontend()
        self.check_backend()
        self.check_ports()
        self.check_configuration()
        
        success = self.print_report()
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        
        if success and not self.errors:
            print("\n1. Start Backend Server:")
            print("   cd backend")
            if sys.platform == 'win32':
                print("   venv\\Scripts\\activate")
            else:
                print("   source venv/bin/activate")
            print("   python app.py")
            
            print("\n2. Start Frontend (in new terminal):")
            print("   cd frontend")
            print("   npm start")
            
            print("\n3. Access Application:")
            print("   Backend:  http://localhost:5000")
            print("   Frontend: http://localhost:3000")
        else:
            print("\nPlease fix the errors above before running the application.")
        
        print("\n" + "="*60)
        
        return success and not self.errors

if __name__ == '__main__':
    checker = VerificationChecker()
    success = checker.run()
    sys.exit(0 if success else 1)
