#!/usr/bin/env python3
import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="samuTrain V2 Server")
  parser.add_argument("--data-folder", default="data", help="Path to data")
  parser.add_argument("--reset-db", action="store_true", help="Reset DB")
  parser.add_argument("--model-folder", default="generic_latin", help="Model subfolder")
  
  args = parser.parse_args()
  
  os.environ['SAMUTRAIN_DATA_FOLDER'] = args.data_folder
  os.environ['SAMUTRAIN_MODEL_NAME'] = args.model_folder
  
  from main import app
  import uvicorn
  from src.db import db
  
  if args.reset_db:
    db.reset_database()
  
  print(f"🚀 Starting with model: {args.model_folder}")
  uvicorn.run(app, host="127.0.0.1", port=8000)