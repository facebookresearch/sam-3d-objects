#!/usr/bin/env python3
"""
Basic validation script to check the backend server structure
without requiring full dependencies to be installed.
"""

import os
import sys
from pathlib import Path

def validate_backend():
    """Validate backend structure and files"""
    print("=== Backend Validation ===\n")
    
    backend_dir = Path(__file__).parent
    errors = []
    warnings = []
    
    # Check required files exist
    required_files = [
        "server.py",
        "requirements.txt",
        "Dockerfile",
        "README.md"
    ]
    
    print("Checking required files...")
    for file in required_files:
        file_path = backend_dir / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            errors.append(f"Missing required file: {file}")
            print(f"  ✗ {file}")
    
    # Check server.py for required endpoints
    print("\nChecking API endpoints...")
    server_path = backend_dir / "server.py"
    if server_path.exists():
        with open(server_path, 'r') as f:
            content = f.read()
        
        endpoints = [
            ("/api/process/image", "@app.post(\"/api/process/image\")"),
            ("/api/process/video", "@app.post(\"/api/process/video\")"),
            ("/api/generate/image-kie", "@app.post(\"/api/generate/image-kie\")"),
            ("/api/generate/3d", "@app.post(\"/api/generate/3d\")"),
            ("/api/health", "@app.get(\"/api/health\")"),
        ]
        
        for endpoint, pattern in endpoints:
            if pattern in content:
                print(f"  ✓ {endpoint}")
            else:
                errors.append(f"Missing endpoint: {endpoint}")
                print(f"  ✗ {endpoint}")
    
    # Check Dockerfile
    print("\nChecking Dockerfile...")
    dockerfile_path = backend_dir / "Dockerfile"
    if dockerfile_path.exists():
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        required_elements = [
            ("CUDA 12.6", "cuda:12.6"),
            ("Python 3.12", "python3.12"),
            ("Environment variables", "ENV HF_TOKEN"),
            ("Port exposure", "EXPOSE 8000"),
        ]
        
        for desc, pattern in required_elements:
            if pattern in dockerfile_content:
                print(f"  ✓ {desc}")
            else:
                warnings.append(f"Dockerfile may be missing: {desc}")
                print(f"  ⚠ {desc}")
    
    # Check requirements.txt
    print("\nChecking requirements.txt...")
    req_path = backend_dir / "requirements.txt"
    if req_path.exists():
        with open(req_path, 'r') as f:
            requirements = f.read()
        
        key_deps = [
            ("FastAPI", "fastapi"),
            ("Uvicorn", "uvicorn"),
            ("PyTorch", "torch"),
            ("Hugging Face Hub", "huggingface-hub"),
        ]
        
        for desc, pattern in key_deps:
            if pattern in requirements:
                print(f"  ✓ {desc}")
            else:
                warnings.append(f"Missing dependency: {desc}")
                print(f"  ⚠ {desc}")
    
    # Check frontend build directory
    print("\nChecking frontend build...")
    frontend_dist = backend_dir.parent / "frontend" / "dist"
    if frontend_dist.exists():
        print(f"  ✓ Frontend build exists at {frontend_dist}")
    else:
        warnings.append("Frontend not built yet - run 'npm run build' in frontend/")
        print(f"  ⚠ Frontend build not found")
    
    # Summary
    print("\n=== Validation Summary ===")
    if errors:
        print(f"\n❌ Found {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print(f"\n⚠ Found {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors:
        print("\n✓ Backend structure validation passed!")
        if warnings:
            print("  Note: Some warnings were found but they may not be critical.")
        return 0
    else:
        print("\n✗ Backend validation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(validate_backend())
