#!/usr/bin/env python3
"""
samuTrain V2 Single Case Test Runner
Complete workflow for testing single_case training
"""

import sys
import os
import subprocess
import time
import requests
from pathlib import Path

def check_server_running():
    """Check if the server is running"""
    try:
        response = requests.get('http://127.0.0.1:8000/api/health', timeout=2)
        return response.status_code == 200
    except:
        return False

def main():
    print("🎯 samuTrain V2 Single Case Test Runner")
    print("=" * 50)
    
    # Configuration
    data_folder = "data/single_case"
    server_url = "http://127.0.0.1:8000"
    
    print(f"📁 Data folder: {data_folder}")
    print(f"🌐 Server URL: {server_url}")
    
    # Check if data folder exists
    if not Path(data_folder).exists():
        print(f"❌ Error: Data folder '{data_folder}' does not exist")
        sys.exit(1)
    
    # Count files
    png_files = list(Path(data_folder).glob("*.png"))
    gt_files = list(Path(data_folder).glob("*.gt.txt"))
    
    print(f"📄 Found {len(png_files)} PNG files")
    print(f"📝 Found {len(gt_files)} GT files")
    
    if not png_files:
        print("❌ No PNG files found in data folder")
        sys.exit(1)
    
    # Check if server is running
    if not check_server_running():
        print("🚀 Starting server...")
        server_proc = subprocess.Popen([
            sys.executable, "run_server.py", 
            "--data-folder", data_folder
        ])
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        for i in range(10):
            time.sleep(1)
            if check_server_running():
                print("✅ Server started successfully")
                break
        else:
            print("❌ Failed to start server")
            sys.exit(1)
    else:
        print("✅ Server already running")
        server_proc = None
    
    try:
        # Run training simulation
        print("\n🎯 Running training simulation...")
        train_proc = subprocess.run([
            sys.executable, "scripts/train.py",
            "--data-folder", data_folder,
            "--epochs", "5"
        ], capture_output=True, text=True)
        
        print("📊 Training output:")
        print(train_proc.stdout)
        
        if train_proc.stderr:
            print("⚠️  Training warnings:")
            print(train_proc.stderr)
        
        # Check results
        print("\n📈 Checking results...")
        try:
            stats = requests.get(f"{server_url}/api/statistics").json()
            cases = requests.get(f"{server_url}/api/cases").json()
            
            print(f"📊 Total cases: {stats.get('total_cases', 0)}")
            print(f"❌ Failset cases: {stats.get('failset_cases', 0)}")
            print(f"📈 Average confidence: {stats.get('avg_confidence', 0):.3f}")
            
            if cases:
                print("\n📋 Case details:")
                for case in cases:
                    status = "FAILSET" if case['is_failset'] else "TRAINSET"
                    if case['is_corrected']:
                        status += " (CORRECTED)"
                    print(f"  🆔 {case['id']}: {case['ocr_text']} - {case['confidence']:.1%} - {status}")
            
        except Exception as e:
            print(f"❌ Error checking results: {e}")
        
        print(f"\n🌐 UI available at: {server_url}")
        print("📚 Open the URL in your browser to see the results")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    finally:
        # Clean up server if we started it
        if server_proc:
            print("\n🛑 Stopping server...")
            server_proc.terminate()
            server_proc.wait()

if __name__ == "__main__":
    main()
