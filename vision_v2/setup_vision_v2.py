"""
Setup Script for Vision System V2.0
Installs all packages with NumPy 2.x compatibility
"""
import subprocess
import sys

def run_command(cmd):
    """Run a pip command and return success status"""
    print(f"\n{'='*70}")
    print(f"Running: {cmd}")
    print('='*70)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    print("\n" + "="*70)
    print("VISION SYSTEM V2.0 - SETUP (NumPy 2.x Compatible)")
    print("="*70)
    print("\nThis will install all required packages compatible with NumPy 2.x")
    print("="*70)
    
    # Check current NumPy version
    try:
        import numpy as np
        numpy_version = np.__version__
        print(f"\n✓ NumPy version: {numpy_version}")
        
        major_version = int(numpy_version.split('.')[0])
        if major_version >= 2:
            print("  Using NumPy 2.x - will install compatible packages")
        else:
            print("  Using NumPy 1.x - consider upgrading to NumPy 2.x")
    except ImportError:
        print("\n⚠ NumPy not found - will be installed")
    
    print("\n" + "="*70)
    print("Installing packages...")
    print("="*70)
    
    # Packages to install with NumPy 2.x compatible versions
    packages = [
        # Core packages - NumPy 2.x compatible versions
        ("numpy>=2.0", "NumPy (keeping 2.x)"),
        ("opencv-python>=4.10.0", "OpenCV (NumPy 2.x compatible)"),
        ("Pillow>=10.0.0", "Pillow (image processing)"),
        ("mss>=9.0.1", "MSS (screen capture)"),
        ("pytesseract>=0.3.10", "PyTesseract (OCR)"),
    ]
    
    results = []
    
    for package, description in packages:
        print(f"\nInstalling {description}...")
        success = run_command(f"pip install {package} --upgrade --break-system-packages")
        results.append((description, success))
    
    # Summary
    print("\n" + "="*70)
    print("INSTALLATION SUMMARY")
    print("="*70 + "\n")
    
    for desc, success in results:
        status = "✓ INSTALLED" if success else "✗ FAILED"
        print(f"{status:15s} - {desc}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nSuccess: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL PACKAGES INSTALLED!")
        print("\nNext steps:")
        print("  1. Install Tesseract OCR (optional but recommended)")
        print("     Download: https://github.com/UB-Mannheim/tesseract/wiki")
        print("\n  2. Test the installation:")
        print("     python test_vision_v2.py")
        print("\n  3. Try screen capture:")
        print("     python screen_capture_v2.py")
    else:
        print("\n⚠ Some packages failed to install")
        print("Try running the failed packages manually:")
        for desc, success in results:
            if not success:
                print(f"  pip install <package-for-{desc}> --break-system-packages")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
