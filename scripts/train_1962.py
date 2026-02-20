#!/usr/bin/env python3

import os
import sys
import subprocess

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def train_model_on_1962():
  """Train a model on the 1962 dataset"""
  print("🎯 Training model on 1962 dataset...")
  
  # Create output directory
  output_dir = "models/model_1962"
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  # Build command
  cmd = [
    sys.executable, "-m", "calamari_ocr.scripts.train",
    "--train.images", "data/1962/*.bin.png",
    "--trainer.epochs", "5",
    "--trainer.output_dir", output_dir,
    "--trainer.best_model_prefix", "best",
    "--trainer.gen", "TrainOnly",
    "--train.batch_size", "1",
    "--train.num_processes", "2",
    "--codec.auto_compute", "True",
    "--trainer.progress_bar", "True"
  ]
  
  print(f"Command: {' '.join(cmd)}")
  
  try:
    # Run command
    result = subprocess.run(cmd, check=True, text=True)
    print("✅ Training completed successfully!")
    return True
    
  except subprocess.CalledProcessError as e:
    print(f"❌ Training failed: {e}")
    print(f"Error output: {e.stderr}")
    return False
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    return False

if __name__ == "__main__":
  success = train_model_on_1962()
  if success:
    print("🎉 Model training complete! You can now update the bridge to use this model.")
  else:
    print("💥 Model training failed!")
