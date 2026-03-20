#!/usr/bin/env python3
"""
samuTrain V2 Server Launcher
Run this script to start the samuTrain OCR monitoring server
"""

import os
import sys
import argparse

# Add src/ and project root to sys.path for proper module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Force use of virtual environment
venv_python = os.path.join(project_root, 'venv_310', 'Scripts', 'python.exe')
if os.path.exists(venv_python) and sys.executable != venv_python:
    print(f"⚠️ Switching to virtual environment Python: {venv_python}")
    os.execv(venv_python, [venv_python] + sys.argv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start samuTrain V2 Server")
    parser.add_argument("--data-folder", default="data/64_case",
                       help="Path to data folder (default: data/64_case)")
    parser.add_argument("--model-folder", default="models/new_model",
                       help="Path to model folder (default: models/new_model)")
    parser.add_argument("--reset-db", action="store_true",
                       help="Reset database on startup")
    
    args = parser.parse_args()

    # Ensure UTF-8 output on Windows (avoids UnicodeEncodeError for emoji in db.py)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    
    # Set data folder as environment variable for other modules
    os.environ['SAMUTRAIN_DATA_FOLDER'] = args.data_folder
    os.environ['SAMUTRAIN_MODEL_FOLDER'] = args.model_folder
    
    from main import app
    import uvicorn
    
    if args.reset_db:
        print("🔄 Resetting database...")
        from src.db import db
        success = db.reset_database()
        if not success:
            print("❌ Database reset failed")
            exit(1)
        print("� Database reset enabled")
    
    print("🔄🚀 Starting samuTrain V2 Server...")
    print(f"📁 Data folder: {args.data_folder}")
    print(f"📁 Model folder: {args.model_folder}")
    if args.reset_db:
        print("🔄 Database reset enabled")
    print("📊 UI will be available at: http://127.0.0.1:8000")
    print("📚 API docs at: http://127.0.0.1:8000/docs")
    print("⏹️  Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False)
