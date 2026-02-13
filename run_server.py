#!/usr/bin/env python3
"""
samuTrain V2 Server Launcher
Run this script to start the samuTrain OCR monitoring server
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    from main import app
    import uvicorn
    
    print("🚀 Starting samuTrain V2 Server...")
    print("📊 UI will be available at: http://127.0.0.1:8000")
    print("📚 API docs at: http://127.0.0.1:8000/docs")
    print("⏹️  Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
