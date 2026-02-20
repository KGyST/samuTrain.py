#!/usr/bin/env python3

import os
import sys
import subprocess

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def train_model_on_dataset(dataset_name, num_processes=2):
  """Train a model on any dataset"""
  print(f"🎯 Training model on {dataset_name} dataset...")
  
  # Create output directory
  output_dir = f"models/model_{dataset_name}"
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  # Build command
  cmd = [
    sys.executable, "-m", "calamari_ocr.scripts.train",
    "--train.images", f"data/{dataset_name}/*.bin.png",
    "--trainer.epochs", "5",
    "--trainer.output_dir", output_dir,
    "--trainer.best_model_prefix", "best",
    "--trainer.gen", "TrainOnly",
    "--train.batch_size", "1",
    "--train.num_processes", str(num_processes),
    "--codec.auto_compute", "True",
    "--trainer.progress_bar", "True"
  ]
  
  print(f"Command: {' '.join(cmd)}")
  
  try:
    # Run command
    result = subprocess.run(cmd, check=True, text=True)
    print("✅ Training completed successfully!")
    return True, output_dir
    
  except subprocess.CalledProcessError as e:
    print(f"❌ Training failed: {e}")
    print(f"Error output: {e.stderr}")
    return False, None
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    return False, None

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python train_dataset.py <dataset_name> [num_processes]")
    print("Example: python train_dataset.py 1962 2")
    sys.exit(1)
  
  dataset_name = sys.argv[1]
  num_processes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
  
  success, model_dir = train_model_on_dataset(dataset_name, num_processes)
  if success:
    print(f"🎉 Model training complete! Model saved to: {model_dir}")
    print(f"💡 To use this model, run server with:")
    print(f"   python scripts/run_server.py --data-folder data/{dataset_name} --model-folder {model_dir}")
  else:
    print("💥 Model training failed!")
