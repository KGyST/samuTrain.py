#!/usr/bin/env python3
import os
import sys
import subprocess
import signal
import time
import shutil
import json
import argparse
from datetime import datetime

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'

# Global variables for graceful shutdown
training_process = None
dest_model = None

def signal_handler(signum, frame):
  """Handle Ctrl+C gracefully and save best model"""
  global training_process, dest_model
  print(f"\n🛑 Received signal {signum}. Saving best model before shutdown...")
  
  if training_process:
    print("🔄 Attempting to save current model as best...")
    
    # Try to send SIGUSR1 to trigger model save (if supported)
    try:
      training_process.send_signal(signal.SIGUSR1)
      print("📤 Sent save signal to training process")
    except (AttributeError, OSError):
      print("⚠️ Cannot send save signal, will terminate gracefully")
    
    print("🔄 Terminating training process...")
    training_process.terminate()
    try:
      # Give more time for potential model saving
      training_process.wait(timeout=60)  # Wait up to 60 seconds
      print("✅ Training terminated gracefully")
    except subprocess.TimeoutExpired:
      print("⚠️ Training didn't terminate gracefully, forcing kill...")
      training_process.kill()
      training_process.wait()
      print("✅ Training force-killed")
  
  print("👋 Exiting...")
  sys.exit(0)

def validate_arguments(train_folder, source_model, dest_model, force_scratch, network):
  """Validate input arguments"""
  
  # Check training folder
  if not os.path.exists(train_folder):
    print(f"❌ Training folder not found: {train_folder}")
    return False
  
  # Check if training folder contains .bin.png files
  png_files = [f for f in os.listdir(train_folder) if f.endswith('.bin.png')]
  if not png_files:
    print(f"❌ No .bin.png files found in training folder: {train_folder}")
    return False
  
  # Check for corresponding .gt.txt files
  missing_gt = []
  for png_file in png_files:
    gt_file = png_file.replace('.bin.png', '.gt.txt')
    if not os.path.exists(os.path.join(train_folder, gt_file)):
      missing_gt.append(gt_file)
  
  if missing_gt:
    print(f"❌ Missing ground truth files: {missing_gt[:5]}{'...' if len(missing_gt) > 5 else ''}")
    return False
  
  print(f"✅ Found {len(png_files)} training images with ground truth")
  
  # Check source model (optional)
  if source_model:
    # Check if it's a directory with best.ckpt.json
    if os.path.isdir(source_model):
      source_json = os.path.join(source_model, "best.ckpt.json")
    else:
      # Check if it's already a path to json file
      source_json = source_model if source_model.endswith('.json') else f"{source_model}.json"
    
    if not os.path.exists(source_json):
      print(f"❌ Source model not found: {source_json}")
      return False
    print(f"✅ Source model found: {source_json}")
  
  # Handle force_scratch logic
  if force_scratch and os.path.exists(dest_model):
    # Check if there's an existing model
    model_json = os.path.join(dest_model, "best.ckpt.json")
    if os.path.exists(model_json):
      # Archive old model to .old/
      old_dir = os.path.join(project_root, ".old")
      if not os.path.exists(old_dir):
        os.makedirs(old_dir)
      
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      archive_name = f"{os.path.basename(dest_model)}_old_{timestamp}"
      archive_path = os.path.join(old_dir, archive_name)
      
      print(f"🗂️ Archiving existing model to: {archive_path}")
      shutil.move(dest_model, archive_path)
      
      # Recreate clean destination folder
      os.makedirs(dest_model)
      print(f"✅ Created clean destination folder: {dest_model}")
    else:
      print(f"⚠️ Destination folder exists but no model found: {dest_model}")
  elif not os.path.exists(dest_model):
    os.makedirs(dest_model)
    print(f"✅ Created destination folder: {dest_model}")
  else:
    print(f"⚠️ Destination folder exists: {dest_model}")
  
  # Validate network parameter
  if not network or not network.strip():
    print(f"❌ Network parameter cannot be empty")
    return False
  
  # Print architecture summary
  if force_scratch:
    print(f"🏗️ Building new model from scratch with architecture: {network}")
  elif source_model:
    print(f"🔄 Fine-tuning model {source_model} with architecture: {network}")
  else:
    print(f"🏗️ Training from scratch with architecture: {network}")
  
  return True

def train_model_continuous(train_folder, source_model, dest_model_param, network, force_scratch):
  """Train model continuously with graceful shutdown support"""
  
  global dest_model
  dest_model = dest_model_param
  
  print(f"🔧 Starting continuous training...")
  print(f"📁 Training data: {train_folder}")
  print(f"📦 Source model: {source_model if source_model else 'None (training from scratch)'}")
  print(f"🎯 Destination: {dest_model}")
  print(f"⚠️ Press Ctrl+C to stop training and save model")
  
  # Build training command with smooth training parameters
  cmd = [
    sys.executable, "-m", "calamari_ocr.scripts.train",
    "--train.images", os.path.join(train_folder, "*.bin.png"),
    "--trainer.epochs", "1000",  # Large number for continuous training
    "--trainer.output_dir", dest_model,
    "--trainer.best_model_prefix", "best",
    "--trainer.gen", "TrainOnly",
    "--train.batch_size", "1",
    "--train.num_processes", str(min(2, os.cpu_count() or 1)),  # Cap at 2 for Windows efficiency
    "--codec.auto_compute", "True",
    "--trainer.progress_bar", "True",
    "--trainer.val_every_n", "10",  # Validate every 10 epochs for smooth training with regular saves
    "--trainer.progress_bar_mode", "0",  # Minimize progress bar output
    "--trainer.tf_cpp_min_log_level", "3",  # Reduce TensorFlow logging
    "--trainer.write_checkpoints", "True",  # Ensure checkpoints are written
    "--network", network,  # Add network architecture
  ]
  
  # Add source model if provided (only if not force_scratch)
  if source_model and not force_scratch:
    # Use the directory path for checkpoint
    checkpoint_path = source_model if os.path.isdir(source_model) else source_model.replace('.json', '')
    cmd.extend(["--checkpoint", checkpoint_path])
  
  print(f"🚀 Training command: {' '.join(cmd)}")
  print(f"⏰ Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
  
  global training_process
  
  try:
    # Start training process
    training_process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      universal_newlines=True,
      bufsize=1
    )
    
    # Monitor output in real-time
    for line in iter(training_process.stdout.readline, ''):
      if line.strip():
        print(f"📊 {line.strip()}")
    
    # Training completed normally
    training_process.wait()
    print("✅ Training completed normally!")
    return True
    
  except KeyboardInterrupt:
    # Handled by signal handler
    return True
  except Exception as e:
    print(f"❌ Training error: {e}")
    return False
  finally:
    training_process = None

if __name__ == "__main__":
  # Set up signal handlers
  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)
  
  # Parse arguments
  parser = argparse.ArgumentParser(description="Train Calamari OCR model with rapid architecture experimentation")
  parser.add_argument("train_folder", help="Training data folder with .bin.png and .gt.txt files")
  parser.add_argument("dest_model", help="Destination model folder")
  parser.add_argument("source_model", nargs="?", help="Source model for fine-tuning (optional)")
  parser.add_argument("--force_scratch", action="store_true", help="Force training from scratch, archive existing model")
  parser.add_argument("--network", default="cnn=16:3x3,pool=2x2,lstm=64,dropout=0.5", 
                     help="Network architecture string (default: cnn=16:3x3,pool=2x2,lstm=64,dropout=0.5)")
  
  args = parser.parse_args()
  
  # Validate force_scratch conflicts
  if args.force_scratch and args.source_model:
    print("❌ Cannot use --force_scratch with source_model. Use either --force_scratch for new training or source_model for fine-tuning.")
    sys.exit(1)
  
  # Validate arguments
  if not validate_arguments(args.train_folder, args.source_model, args.dest_model, args.force_scratch, args.network):
    sys.exit(1)
  
  # Start training
  success = train_model_continuous(args.train_folder, args.source_model, args.dest_model, args.network, args.force_scratch)
  
  if success:
    print(f"\n✅ Training completed! Model saved to: {args.dest_model}")
    print(f"🧪 Test with: python try_test_model.py {args.dest_model} <image_path>")
  else:
    print(f"\n❌ Training failed!")
    sys.exit(1)
