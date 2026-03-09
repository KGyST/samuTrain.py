# What is this file for: Script to continue learning using Calamari train script with checkpoint
# When it is created: 2026-03-09

import os
import sys
import subprocess
import json
import shutil
from datetime import datetime

# Force minimal logging at the OS level (3 = ERROR only)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

def collect_chars(data_folder):
  """Collect unique characters from ground truth files"""
  chars = set()
  for root, dirs, files in os.walk(data_folder):
    for file in files:
      if file.endswith('.gt.txt'):
        gt_file = os.path.join(root, file)
        try:
          with open(gt_file, 'r', encoding='utf-8') as f:
            chars.update(f.read())
        except:
          pass
  return sorted(list(chars))

def backup_old_model(model_dir):
  """Backup old model directory with timestamp"""
  if not os.path.exists(model_dir):
    return None
  
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  backup_dir = f"{model_dir}_backup_{timestamp}"
  
  try:
    shutil.copytree(model_dir, backup_dir)
    print(f"Old model backed up to: {backup_dir}")
    return backup_dir
  except Exception as e:
    print(f"Warning: Could not backup old model: {e}")
    return None

def train_with_checkpoint(data_folder, old_model_dir, new_model_dir=None, replace_old=False):
  """Train new model using checkpoint from old model"""
  
  # Validate paths
  if not os.path.exists(old_model_dir):
    raise FileNotFoundError(f"Old model directory not found: {old_model_dir}")
  
  if not os.path.exists(data_folder):
    raise FileNotFoundError(f"Data directory not found: {data_folder}")
  
  # Find checkpoint file
  checkpoint_file = os.path.join(old_model_dir, "best.ckpt")
  if not os.path.exists(checkpoint_file):
    checkpoint_file = os.path.join(old_model_dir, "best.ckpt.json")
  
  if not os.path.exists(checkpoint_file):
    raise FileNotFoundError(f"Checkpoint not found in: {old_model_dir}")
  
  # Determine output directory
  if new_model_dir is None:
    if replace_old:
      # Backup old model and use same directory
      backup_old_model(old_model_dir)
      new_model_dir = old_model_dir
    else:
      # Create new directory name
      base_name = old_model_dir.rstrip('/')
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      new_model_dir = f"{base_name}_extended_{timestamp}"
  
  # Collect characters for verification
  chars = collect_chars(data_folder)
  print(f"Characters in new data: {''.join(chars)}")
  print(f"Total unique characters: {len(chars)}")
  
  # Construct training command using correct parameter names
  cmd = [
    sys.executable, "-m", "calamari_ocr.scripts.train",
    "--warmstart.model", checkpoint_file,
    "--trainer.auto_upgrade_checkpoints", "True",
    "--trainer.output_dir", new_model_dir,
    "--train.images", os.path.join(data_folder, "*.bin.png"),
    "--train.skip_invalid", "True",
    "--train.batch_size", "1",  # Use batch size 1 for small datasets
    "--trainer.epochs", "100",
    "--trainer.gen", "TrainOnly",  # Use training data for validation
    "--warmstart.allow_partial", "True",  # Allow partial weight loading
    "--network", "cnn=8:3x3,pool=2x2,lstm=32"  # Use same network as original model
  ]
  
  print(f"Training command:")
  print(" ".join(f'"{arg}"' if " " in arg else arg for arg in cmd))
  print(f"New model will be created in: {new_model_dir}")
  
  # Run training
  try:
    result = subprocess.run(cmd, check=True)
    print(f"Training completed successfully!")
    print(f"New model location: {new_model_dir}")
    
    # Show final codec if available
    final_params = os.path.join(new_model_dir, "trainer_params.json")
    if os.path.exists(final_params):
      with open(final_params, 'r') as f:
        params = json.load(f)
      if "scenario" in params and "data" in params["scenario"] and "codec" in params["scenario"]["data"]:
        final_charset = params["scenario"]["data"]["codec"]["charset"]
        print(f"Final model charset: {''.join(final_charset)}")
        print(f"Final model classes: {len(final_charset)}")
    
    return new_model_dir
    
  except subprocess.CalledProcessError as e:
    print(f"Training failed with exit code: {e.returncode}")
    raise
  except Exception as e:
    print(f"Unexpected error during training: {e}")
    raise

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python try_train_with_checkpoint.py <data_folder> <old_model_dir> [new_model_dir]")
    print("  data_folder: Path to new training data with .bin.png and .gt.txt files")
    print("  old_model_dir: Path to existing model directory")
    print("  new_model_dir: Optional path for new model (default: creates timestamped directory)")
    print("")
    print("Examples:")
    print("  python try_train_with_checkpoint.py data/single_case models/tiny_test")
    print("  python try_train_with_checkpoint.py data/single_case models/tiny_test models/extended_model")
    sys.exit(1)
  
  data_folder = sys.argv[1]
  old_model_dir = sys.argv[2]
  new_model_dir = sys.argv[3] if len(sys.argv) > 3 else None
  
  try:
    train_with_checkpoint(data_folder, old_model_dir, new_model_dir)
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
