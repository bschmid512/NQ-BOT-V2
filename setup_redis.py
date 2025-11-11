#!/usr/bin/env python3
import os
import sys
import subprocess
import platform

def setup_redis():
    """Setup Redis for NQ-BOT-V2 enhancements"""
    system = platform.system()
    
    if system == "Linux":
        print("Attempting to install Redis on Linux...")
        try:
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "redis-server"], check=True)
            subprocess.run(["sudo", "systemctl", "start", "redis-server"], check=True)
            subprocess.run(["sudo", "systemctl", "enable", "redis-server"], check=True)
            print("[OK] Redis installed and started on Linux.")
        except Exception as e:
            print("[FAIL] Linux Redis installation failed: {}. Please install manually.".format(e))
            return False
            
    elif system == "Darwin":  # macOS
        print("Attempting to install Redis on macOS via Homebrew...")
        try:
            subprocess.run(["brew", "install", "redis"], check=True)
            subprocess.run(["brew", "services", "start", "redis"], check=True)
            print("[OK] Redis installed and started on macOS.")
        except Exception as e:
            print("[FAIL] macOS Redis installation failed: {}. Is Homebrew installed? Please install manually.".format(e))
            return False
            
    elif system == "Windows":
        print("[INFO] Redis on Windows is best run via WSL (Windows Subsystem for Linux).")
        print("This script cannot automate this process. Please follow these steps manually:")
        print("1. Open PowerShell as Administrator and run: wsl --install")
        print("2. After setup, open your WSL terminal (e.g., 'Ubuntu') and run:")
        print("   sudo apt update && sudo apt install -y redis-server")
        print("   sudo service redis-server start")
        print("[INFO] Alternatively, download the latest .msi from:")
        print("https://github.com/microsoftarchive/redis/releases")
        print("[WARN] Please install Redis manually and then re-run this integration script.")
        return False  # Return False because the script didn't complete the install

    else:
        print("[WARN] Unsupported OS: {}. Please install Redis manually.".format(system))
        return False
    
    # Test Redis connection
    print("Testing Redis connection...")
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("[OK] Redis connection successful")
        return True
    except Exception as e:
        print("[FAIL] Redis connection failed: {}".format(e))
        print("[INFO] Ensure Redis is running and accessible at localhost:6379")
        return False

if __name__ == "__main__":
    if not setup_redis():
        sys.exit(1)
