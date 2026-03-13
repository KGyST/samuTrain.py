# What is this file for: Finalized continue learning script with codec extension, network resizing, and graceful shutdown
# When it is created: 2026-03-09 (consolidated from multiple implementations)

import os
import sys
import subprocess
import json
import shutil
import signal
import argparse
from datetime import datetime

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Force minimal logging at the OS level (3 = ERROR only)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'

# Set UTF-8 encoding for stdout to handle emoji characters
if sys.platform == "win32":
  import codecs
  sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

# Global variables for graceful shutdown
training_process = None
model_dir = None

def clean_unicode_text(text):
  """Remove or replace invisible Unicode control characters for better display"""
  if not text:
    return text
  
  # Handle both actual Unicode characters and their escaped representations
  replacements = {
    # Actual Unicode characters
    '\u200b': '',  # Zero Width Space
    '\u200c': '',  # Zero Width Non-Joiner
    '\u200d': '',  # Zero Width Joiner
    '\u200e': '',  # Left-to-Right Mark
    '\u200f': '',  # Right-to-Left Mark
    '\u202a': '[LTR]',  # Left-to-Right Embedding
    '\u202b': '[RTL]',  # Right-to-Left Embedding
    '\u202c': '[PDF]',  # Pop Directional Formatting
    '\u202d': '[LRO]',  # Left-to-Right Override
    '\u202e': '[RLO]',  # Right-to-Left Override
    '\u2060': '',  # Word Joiner
    '\u2061': '',  # Function Application
    '\u2062': '',  # Invisible Separator
    '\u2063': '',  # Invisible Plus
    '\u2064': '',  # Invisible Times
    '\ufeff': '',  # Zero Width No-Break Space (BOM)
    
    # Escaped string representations
    '\\u200b': '',  # Zero Width Space
    '\\u200c': '',  # Zero Width Non-Joiner
    '\\u200d': '',  # Zero Width Joiner
    '\\u200e': '',  # Left-to-Right Mark
    '\\u200f': '',  # Right-to-Left Mark
    '\\u202a': '[LTR]',  # Left-to-Right Embedding
    '\\u202b': '[RTL]',  # Right-to-Left Embedding
    '\\u202c': '[PDF]',  # Pop Directional Formatting
    '\\u202d': '[LRO]',  # Left-to-Right Override
    '\\u202e': '[RLO]',  # Right-to-Left Override
    '\\u2060': '',  # Word Joiner
    '\\u2061': '',  # Function Application
    '\\u2062': '',  # Invisible Separator
    '\\u2063': '',  # Invisible Plus
    '\\u2064': '',  # Invisible Times
    '\\ufeff': '',  # Zero Width No-Break Space (BOM)
  }
  
  # Replace control characters with readable alternatives or remove them
  cleaned = text
  for char, replacement in replacements.items():
    cleaned = cleaned.replace(char, replacement)
  
  return cleaned

def collect_chars(data_folder):
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

def backup_model(model_dir):
  """Backup model directory by renaming to .old format"""
  if not os.path.exists(model_dir):
    return None
  
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  backup_dir = f"{model_dir}.{timestamp}.old"
  
  try:
    shutil.move(model_dir, backup_dir)
    print(f"Model backed up to: {backup_dir}")
    return backup_dir
  except Exception as e:
    print(f"Warning: Could not backup model: {e}")
    return None

def get_current_network(checkpoint_folder):
  """Extract current network architecture from trainer params"""
  params_path = os.path.join(checkpoint_folder, "trainer_params.json")
  if not os.path.exists(params_path):
    return "cnn=8:3x3,pool=2x2,lstm=32"  # Default network
  
  try:
    with open(params_path, 'r') as f:
      params = json.load(f)
    return params.get("network", "cnn=8:3x3,pool=2x2,lstm=32")
  except:
    return "cnn=8:3x3,pool=2x2,lstm=32"

def signal_handler(signum, frame):
  """Handle Ctrl+C gracefully and save best model"""
  global training_process, model_dir
  print(f"\nReceived signal {signum}. Saving best model before shutdown...")
  
  if training_process:
    print("Attempting to save current model as best...")
    
    # Try to send SIGUSR1 to trigger model save (if supported)
    try:
      training_process.send_signal(signal.SIGUSR1)
      print("Sent save signal to training process")
    except (AttributeError, OSError):
      print("Cannot send save signal, will terminate gracefully")
    
    print("Terminating training process...")
    training_process.terminate()
    try:
      # Give more time for potential model saving
      training_process.wait(timeout=60)  # Wait up to 60 seconds
      print("Training terminated gracefully")
    except subprocess.TimeoutExpired:
      print("Training didn't terminate gracefully, forcing kill...")
      training_process.kill()
      training_process.wait()
      print("Training force-killed")
  
  print("Exiting...")
  sys.exit(0)

def copy_essential_files(src_dir, dst_dir, files_to_copy=None):
  """Copy essential files from backup to new model folder for continuation"""
  if files_to_copy is None:
    files_to_copy = [
      "best.ckpt", "best.ckpt.json", "best.ckpt.data-00000-of-00001", "best.ckpt.index",
      "trainer_params.json", "extended_charset.txt", "charset.txt"
    ]
  
  copied_files = []
  for filename in files_to_copy:
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)
    if os.path.exists(src_path):
      try:
        if os.path.isdir(src_path):
          # Copy directory recursively
          if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
          shutil.copytree(src_path, dst_path)
          copied_files.append(filename + " (directory)")
          print(f"Copied essential directory: {filename}")
        elif os.path.isfile(src_path):
          shutil.copy2(src_path, dst_path)
          copied_files.append(filename)
          print(f"Copied essential file: {filename}")
      except Exception as e:
        print(f"Warning: Could not copy {filename}: {e}")
  
  if copied_files:
    print(f"Copied {len(copied_files)} essential files/directories for model continuation")
  else:
    print("Warning: No essential files found to copy")

def continue_learning(data_folder, checkpoint_folder, network=None, backup=True):
  """Continue learning with codec extension, network resizing, and graceful shutdown"""
  global model_dir
  model_dir = checkpoint_folder

  # Validate inputs
  if not os.path.exists(data_folder):
    raise FileNotFoundError(f"Data directory not found: {data_folder}")

  if not os.path.exists(checkpoint_folder):
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_folder}")

  # Pre-training backup: rename original folder to .old
  backup_dir = None
  if backup:
    backup_dir = backup_model(checkpoint_folder)
    if backup_dir is None:
      print("Warning: Backup failed, proceeding without backup")
    else:
      # Create new empty folder for training
      os.makedirs(checkpoint_folder)
      # Copy essential files from backup to new folder
      copy_essential_files(backup_dir, checkpoint_folder)

  # Find checkpoint file in original directory first
  checkpoint_file = os.path.join(checkpoint_folder, "best.ckpt")
  if not os.path.exists(checkpoint_file):
    checkpoint_file = os.path.join(checkpoint_folder, "best.ckpt.json")

  if not os.path.exists(checkpoint_file):
    raise FileNotFoundError(f"Checkpoint not found in: {checkpoint_folder}")

  # Train to new folder
  new_model_dir = checkpoint_folder
  
  # Collect characters and determine network architecture
  chars = collect_chars(data_folder)
  print(f"Characters in new data: {''.join(chars)}")
  print(f"Total unique characters: {len(chars)}")
  
  # Determine network architecture
  if network is None:
    current_network = get_current_network(checkpoint_folder)
    network = current_network
    print(f"Using current network: {network}")
  else:
    print(f"Using specified network: {network}")
  
  # Collect characters for codec extension
  if chars:
    # Create charset file for include_files - this is the proper Calamari way
    charset_file = os.path.join(checkpoint_folder, "extended_charset.txt")
    with open(charset_file, 'w', encoding='utf-8') as f:
      f.write(''.join(chars))
    print(f"Extended charset file created: {charset_file}")
  
  # Construct training command using correct parameter names
  cmd = [
    sys.executable, "-m", "calamari_ocr.scripts.train",
    "--warmstart.model", checkpoint_file,
    "--trainer.auto_upgrade_checkpoints", "True",
    "--trainer.output_dir", new_model_dir,
    "--train.images", os.path.join(data_folder, "*.bin.png"),
    "--train.skip_invalid", "True",
    "--train.batch_size", "1",  # Use batch size 1 for small datasets
    "--trainer.gen", "TrainOnly",  # Use training data for validation
    "--early_stopping.n_to_go", "-1",  # Disable early stopping
    "--network", network
  ]
  
  # Add codec extension parameters if we have new characters
  if chars:
    cmd.extend([
      "--codec.include_files", charset_file,
      "--codec.auto_compute", "True",
      "--codec.keep_loaded", "True"
    ])
  
  print(f"Training command:")
  print(" ".join(f'"{arg}"' if " " in arg else arg for arg in cmd))
  print(f"New model will be created in: {new_model_dir}")
  
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
        cleaned_line = clean_unicode_text(line.strip())
        print(f"{cleaned_line}")
    
    # Training completed normally
    training_process.wait()
    print("Training completed successfully!")
    
    # Show final codec if available
    final_params = os.path.join(checkpoint_folder, "trainer_params.json")
    if os.path.exists(final_params):
      with open(final_params, 'r') as f:
        params = json.load(f)
      if "scenario" in params and "data" in params["scenario"] and "codec" in params["scenario"]["data"]:
        final_charset = params["scenario"]["data"]["codec"]["charset"]
        print(f"Final model charset: {''.join(final_charset)}")
        print(f"Final model classes: {len(final_charset)}")
    
    return checkpoint_folder
    
  except KeyboardInterrupt:
    # Handled by signal handler
    return new_model_dir
  except subprocess.CalledProcessError as e:
    print(f"Training failed with exit code: {e.returncode}")
    raise
  except Exception as e:
    print(f"Unexpected error during training: {e}")
    raise
  finally:
    training_process = None

if __name__ == "__main__":
  # Set up signal handlers for graceful shutdown
  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)
  
  # Parse arguments
  parser = argparse.ArgumentParser(description="Continue learning with codec extension, network resizing, and graceful shutdown")
  parser.add_argument("data_folder", help="Path to new training data with .bin.png and .gt.txt files")
  parser.add_argument("checkpoint_folder", help="Path to existing model checkpoint directory")
  parser.add_argument("--network", help="Network architecture string (default: auto-detect based on data)")
  parser.add_argument("--no-backup", action="store_true", help="Skip model backup before training")
  
  args = parser.parse_args()
  
  try:
    result = continue_learning(
      args.data_folder, 
      args.checkpoint_folder, 
      network=args.network,
      backup=not args.no_backup
    )
    print(f"\nContinue learning completed! Model replaced at original location: {args.checkpoint_folder}")
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
