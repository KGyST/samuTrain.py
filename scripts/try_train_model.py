#!/usr/bin/env python3
import os
import sys
import subprocess
import signal
import time

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'

# Global variable for graceful shutdown
training_process = None

def signal_handler(signum, frame):
  """Handle Ctrl+C gracefully"""
  global training_process
  print(f"\n🛑 Received signal {signum}. Shutting down training gracefully...")
  
  if training_process:
    print("🔄 Terminating training process...")
    training_process.terminate()
    try:
      training_process.wait(timeout=30)  # Wait up to 30 seconds for graceful shutdown
      print("✅ Training terminated gracefully")
    except subprocess.TimeoutExpired:
      print("⚠️ Training didn't terminate gracefully, forcing kill...")
      training_process.kill()
      training_process.wait()
      print("✅ Training force-killed")
  
  print("👋 Exiting...")
  sys.exit(0)

def validate_arguments(train_folder, source_model, dest_model):
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
  
  # Create destination model folder
  if not os.path.exists(dest_model):
    os.makedirs(dest_model)
    print(f"✅ Created destination folder: {dest_model}")
  else:
    print(f"⚠️ Destination folder exists: {dest_model}")
  
  return True

def train_model_continuous(train_folder, source_model, dest_model):
  """Train model continuously with graceful shutdown support"""
  
  print(f"🔧 Starting continuous training...")
  print(f"📁 Training data: {train_folder}")
  print(f"📦 Source model: {source_model if source_model else 'None (training from scratch)'}")
  print(f"🎯 Destination: {dest_model}")
  print(f"⚠️ Press Ctrl+C to stop training and save model")
  
  # Build training command
  cmd = [
    sys.executable, "-m", "calamari_ocr.scripts.train",
    "--train.images", os.path.join(train_folder, "*.bin.png"),
    "--trainer.epochs", "1000",  # Large number for continuous training
    "--trainer.output_dir", dest_model,
    "--trainer.best_model_prefix", "best",
    "--trainer.gen", "TrainOnly",
    "--train.batch_size", "1",
    "--train.num_processes", "1",
    "--codec.auto_compute", "True",
    "--trainer.progress_bar", "True",
    "--trainer.val_every_n", "5",  # Validate every 5 epochs
  ]
  
  # Add source model if provided
  if source_model:
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
  
  if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("Usage: python try_train_model.py <train_folder> <dest_model> [source_model]")
    print("Example: python try_train_model.py data/1962 models/fixed_model")
    print("Example: python try_train_model.py data/1962 models/fixed_model models/generic_latin")
    sys.exit(1)
  
  train_folder = sys.argv[1]
  dest_model = sys.argv[2]
  source_model = sys.argv[3] if len(sys.argv) == 4 else None
  
  # Validate arguments
  if not validate_arguments(train_folder, source_model, dest_model):
    sys.exit(1)
  
  # Start training
  success = train_model_continuous(train_folder, source_model, dest_model)
  
  if success:
    print(f"\n✅ Training completed! Model saved to: {dest_model}")
    print(f"🧪 Test with: python try_test_model.py {dest_model} <image_path>")
  else:
    print(f"\n❌ Training failed!")
    sys.exit(1)
