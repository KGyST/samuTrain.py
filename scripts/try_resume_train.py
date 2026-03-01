import os
import subprocess
import sys

# Use few comments. 2-space indentation. English only.
def resume_training(model_dir, data_dir, output_dir):
  # Get the path of the current python executable from the venv
  current_python = sys.executable
  
  # Use the proper resume_training script that works
  cmd = [
    current_python, "-m", "calamari_ocr.scripts.resume_training",
    model_dir  # Pass model directory, not checkpoint prefix
  ]
  
  print(f"🚀 Resuming training from: {model_dir}")
  print(f"🐍 Using python: {current_python}")
  print(f"📁 Data directory: {data_dir}")
  print(f"📂 Output directory: {output_dir}")
  
  # Forward environment to ensure TF/Calamari is found
  subprocess.run(cmd, env=os.environ.copy())

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python try_resume_train.py <model_directory>")
    print("Example: python try_resume_train.py models/test_zero_interruption_v2")
    sys.exit(1)
    
  model_dir = sys.argv[1]
  # Assuming data is in the default location
  resume_training(model_dir, "data/1962", "models/resumed_model")