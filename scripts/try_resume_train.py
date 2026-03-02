import os
import subprocess
import sys

# Use few comments. 2-space indentation. English only.
def resume_training(data_folder, model_folder):
  # Get the path of the current python executable from the venv
  current_python = sys.executable
  
  # Use the proper resume_training script that works
  cmd = [
    current_python, "-m", "calamari_ocr.scripts.resume_training",
    model_folder  # Pass model directory, not checkpoint prefix
  ]
  
  print(f"🚀 Resuming training from: {model_folder}")
  print(f"🐍 Using python: {current_python}")
  print(f"📁 Data folder: {data_folder}")
  
  # Forward environment to ensure TF/Calamari is found
  subprocess.run(cmd, env=os.environ.copy())

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python try_resume_train.py <data_folder> <model_folder>")
    print("Example: python try_resume_train.py data/1962 models/test_zero_interruption_v2")
    sys.exit(1)
    
  data_folder = sys.argv[1]
  model_folder = sys.argv[2]
  resume_training(data_folder, model_folder)