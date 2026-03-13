# What is this file for: Start training from scratch, then optionally continue training
# When it is created: 2026-03-13 (combines train.py and try_continue_learning.py)

import os
import sys
import json
import subprocess
import argparse
import glob
import signal
from datetime import datetime

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Force minimal logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'

# Protocol Buffers compatibility
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Global variables for graceful shutdown
training_process = None

def normalize_data_path(data_path):
  """Convert folder paths to glob patterns if needed."""
  # If already contains glob pattern or .bin.png, return as-is
  if "*" in data_path or data_path.endswith(".bin.png"):
    return data_path
  
  # If it's a directory, add glob pattern
  if os.path.isdir(data_path):
    if not data_path.endswith(os.sep):
      data_path += os.sep
    return data_path + "*.bin.png"
  
  # If it looks like a directory path (ends with /), add glob pattern
  if data_path.endswith("/"):
    return data_path + "*.bin.png"
  
  # Otherwise, assume it's a glob pattern
  return data_path

def signal_handler(signum, frame):
  """Handle Ctrl+C gracefully"""
  global training_process
  print(f"\n🛑 Received signal {signum}. Terminating training...")
  
  if training_process:
    training_process.terminate()
    try:
      training_process.wait(timeout=30)
      print("✅ Training terminated gracefully")
    except subprocess.TimeoutExpired:
      training_process.kill()
      training_process.wait()
      print("⚠️ Training force-killed")

  sys.exit(0)

def verify_dataset(data_pattern):
  """Verify that dataset exists."""
  images = glob.glob(data_pattern)
  if not images:
    print(f"❌ No images found matching: {data_pattern}")
    return False
  
  for img in images:
    gt_file = img.replace(".bin.png", ".gt.txt")
    if not os.path.exists(gt_file):
      print(f"❌ Missing Ground Truth file: {gt_file}")
      return False
  return True

def start_initial_training(data_pattern, epochs, output_dir, network):
  """Start training from scratch using train.py logic"""
  global training_process
  
  print("🚀 Starting initial training from scratch...")
  print(f"📊 Data: {data_pattern}")
  print(f"🎯 Epochs: {epochs}")
  print(f"📁 Output: {output_dir}")
  
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  # Convert to absolute path
  data_pattern = os.path.abspath(data_pattern)
  output_dir = os.path.abspath(output_dir)
  
  # Build training command (no warmstart for new models)
  cmd = [
    sys.executable, "-m", "calamari_ocr.scripts.train",
    "--train.images", data_pattern,
    "--trainer.epochs", str(epochs),
    "--trainer.output_dir", output_dir,
    "--trainer.best_model_prefix", "best",
    "--trainer.gen", "TrainOnly",
    "--train.batch_size", "1",
    "--val.batch_size", "1",
    "--codec.auto_compute", "True",
    "--network", network
  ]
  
  print(f"Command: {' '.join(cmd)}\n")
  
  try:
    training_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(training_process.stdout.readline, ''):
      if line.strip():
        print(line.strip())
    
    training_process.wait()
    if training_process.returncode == 0:
      print("✅ Initial training completed successfully!")
      return True
    else:
      print(f"❌ Initial training failed with code: {training_process.returncode}")
      return False
      
  except KeyboardInterrupt:
    signal_handler(signal.SIGINT, None)
  except Exception as e:
    print(f"❌ Error during initial training: {e}")
    return False
  finally:
    training_process = None

def continue_training_with_script(model_dir, continue_data, network):
  """Continue training using try_continue_learning.py"""
  print(f"\n🔄 Continuing training with additional data...")
  print(f"📁 Model: {model_dir}")
  print(f"📊 Continue data: {continue_data}")
  print(f"🧠 Network: {network}")
  
  if not verify_dataset(continue_data):
    print("❌ Continue data verification failed")
    return False
  
  # Update trainer_params.json with the correct network
  params_file = os.path.join(model_dir, "trainer_params.json")
  if os.path.exists(params_file):
    try:
      with open(params_file, 'r') as f:
        params = json.load(f)
      params["network"] = network
      with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
      print(f"✅ Updated network in trainer_params.json: {network}")
    except Exception as e:
      print(f"⚠️ Could not update trainer_params.json: {e}")
  
  # Call try_continue_learning.py
  continue_script = os.path.join(script_dir, "try_continue_learning.py")
  # Extract folder path from glob pattern for continuation script
  if "*" in continue_data:
    continue_folder = continue_data.replace("*.bin.png", "").rstrip("/\\")
  else:
    continue_folder = continue_data
  
  cmd = [
    sys.executable, continue_script,
    continue_folder,
    model_dir
  ]
  
  print(f"Continue command: {' '.join(cmd)}\n")
  
  try:
    subprocess.run(cmd, check=True)
    print("✅ Continuation training completed successfully!")
    return True
  except subprocess.CalledProcessError as e:
    print(f"❌ Continuation training failed: {e}")
    return False

def main():
  # Set up signal handlers
  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)
  
  parser = argparse.ArgumentParser(description="Train or continue OCR model")
  parser.add_argument("data_folder", help="Path to training data folder")
  parser.add_argument("model_folder", help="Path to model folder (existing for continuation, new for --new)")
  parser.add_argument("--new", action="store_true", help="Create new model (model_folder must not exist)")
  parser.add_argument("--epochs", type=int, default=5, help="Training epochs for new model")
  parser.add_argument("--network", default="cnn=8:3x3,pool=2x2,lstm=32", help="Network architecture")
  parser.add_argument("--auto-continue", action="store_true", help="Automatically continue with same data after initial training")
  parser.add_argument("--force", action="store_true", help="Force overwrite existing model directory without prompt")
  
  args = parser.parse_args()
  
  # Normalize data path
  data_path = normalize_data_path(args.data_folder)
  
  # Verify dataset
  if not verify_dataset(data_path):
    sys.exit(1)
  
  # Check model folder existence
  model_exists = os.path.exists(args.model_folder)
  
  # Validate arguments
  if args.new and model_exists:
    if not args.force:
      print(f"❌ Model folder already exists: {args.model_folder}")
      print(f"   Use --force to overwrite or choose a different folder")
      sys.exit(1)
    shutil.rmtree(args.model_folder)
    print(f"🗑️ Removed existing model folder: {args.model_folder}")
    model_exists = False
  
  if not args.new and not model_exists:
    print(f"❌ Model folder does not exist: {args.model_folder}")
    print(f"   Use --new to create a new model or provide an existing model folder")
    sys.exit(1)
  
  # Handle new model creation
  if args.new:
    print(f"🆕 Creating new model: {args.model_folder}")
    if not start_initial_training(data_path, args.epochs, args.model_folder, args.network):
      print("❌ Initial training failed")
      sys.exit(1)
    
    # Handle auto-continuation for new models
    if args.auto_continue:
      print("\n🤖 Auto-continuation enabled, proceeding...")
      success = continue_training_with_script(args.model_folder, data_path, args.network)
      if success:
        print(f"\n🎉 Complete training workflow finished! Model: {args.model_folder}")
      else:
        print("❌ Continuation failed")
        sys.exit(1)
    else:
      print(f"\n🎉 New model created! Model saved to: {args.model_folder}")
  
  # Handle model continuation
  else:
    print(f"� Continuing training on existing model: {args.model_folder}")
    success = continue_training_with_script(args.model_folder, data_path, args.network)
    if success:
      print(f"\n🎉 Model continuation finished! Model: {args.model_folder}")
    else:
      print("❌ Continuation failed")
      sys.exit(1)

if __name__ == "__main__":
  main()
