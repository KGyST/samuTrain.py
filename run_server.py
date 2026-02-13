#!/usr/bin/env python3
"""
samuTrain V2 Server Launcher
Run this script to start the samuTrain OCR monitoring server
"""

import sys
import os
import argparse

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start samuTrain V2 Server")
    parser.add_argument("--data-folder", default="data",
                       help="Path to data folder (default: data)")
    
    args = parser.parse_args()
    
    # Set data folder as environment variable for other modules
    os.environ['SAMUTRAIN_DATA_FOLDER'] = args.data_folder
    
    from main import app
    import uvicorn
    
    print("🚀 Starting samuTrain V2 Server...")
    print(f"📁 Data folder: {args.data_folder}")
    print("📊 UI will be available at: http://127.0.0.1:8000")
    print("📚 API docs at: http://127.0.0.1:8000/docs")
    print("⏹️  Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
