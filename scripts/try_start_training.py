# What is this file for: Unified training script that starts from scratch or continues existing models, with checkpoint validation and proper file handling for continuation training. Consolidated from try_continue_learning.py with all functions in one module, simplified logic, and fixed checkpoint copying to extract files to project root instead of creating/overwriting fake checkpoints. Uses last valid checkpoint with enhanced validation.
# When it is created: 2026-03-17 (complete refactoring)

import os, sys, json, subprocess, argparse, glob, shutil, signal, codecs, uuid
from datetime import datetime

from numpy.ma.core import bool_


UTF_8 = 'utf-8'
TRAINER_PARAMS = "trainer_params.json"
BEST_CKPT = "best.ckpt.json"

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

# Set UTF-8 encoding for stdout to handle emoji characters
if sys.platform == "win32":
  import codecs
  sys.stdout = codecs.getwriter(UTF_8)(sys.stdout.detach())

# Global variables for graceful shutdown
training_process = None
model_dir = None

# Import Calamari library components for library mode
try:
  from calamari_ocr.ocr.scenario import CalamariScenario
  from calamari_ocr.scripts.train import main as calamari_train
  LIB_MODE = True
except ImportError:
  print("⚠️ Calamari library not available, falling back to CLI mode")
  LIB_MODE = False

def normalize_data_path(data_path):
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

def clean_unicode_text(text: str) -> str:
  """Remove or replace invisible Unicode control characters for better display"""
  if not text:
    return text
  
  # Handle both actual Unicode characters and their escaped representations
  replacements = {
    # Actual Unicode characters
    # '\u200b': '',  # Zero Width Space
    # '\u200c': '',  # Zero Width Non-Joiner
    # '\u200d': '',  # Zero Width Joiner
    # '\u200e': '',  # Left-to-Right Mark
    # '\u200f': '',  # Right-to-Left Mark
    '\u202a': '[LTR]',  # Left-to-Right Embedding
    '\u202b': '[RTL]',  # Right-to-Left Embedding
    '\u202c': '[PDF]',  # Pop Directional Formatting
    '\u202d': '[LRO]',  # Left-to-Right Override
    '\u202e': '[RLO]',  # Right-to-Left Override
    # '\u2060': '',  # Word Joiner
    # '\u2061': '',  # Function Application
    # '\u2062': '',  # Invisible Separator
    # '\u2063': '',  # Invisible Plus
    # '\u2064': '',  # Invisible Times
    # '\ufeff': '',  # Zero Width No-Break Space (BOM)
  }

  replacements = {
    **replacements,
    **{f'{repr(r)[1:-1]}': s for r, s in replacements.items()}
  }
  
  # Replace control characters with readable alternatives or remove them
  cleaned = text
  for char, replacement in replacements.items():
    cleaned = cleaned.replace(char, replacement)
  
  return cleaned

def collect_chars(data_folder: str):
  chars = set()
  for root, dirs, files in os.walk(data_folder):
    for file in files:
      if file.endswith('.gt.txt'):
        gt_file = os.path.join(root, file)
        try:
          with open(gt_file, 'r', encoding=UTF_8) as f:
            chars.update(f.read())
        except:
          pass
  return sorted(list(chars))

def backup_model(model_dir: str) -> None | str:
  """Backup model directory by renaming to .old format"""
  
  assert os.path.isdir(model_dir)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  backup_dir = f"{model_dir}.{timestamp}.old"

  assert not os.path.exists(backup_dir)
  
  try:
    shutil.move(model_dir, backup_dir)
    print(f"Model backed up to: {backup_dir}")
    return backup_dir
  except Exception as e:
    print(f"Warning: Could not backup model: {e}")
    return None

def get_current_network(model_dir: str) -> None | str:
  """Extract current network architecture from trainer params"""
  
  assert os.path.isdir(model_dir)

  try:
    with open(os.path.join(model_dir, TRAINER_PARAMS), 'r') as f:
      params = json.load(f)
    if network := params["network"]:
      return network
  except:
    pass

  try:
    with open(os.path.join(get_last_valid_checkpoint(model_dir), TRAINER_PARAMS), 'r') as f:
      params = json.load(f)
    if network := params["network"]:
      return network
  except:
    pass

  return None

def signal_handler(signum):
  """Handle Ctrl+C gracefully and save best model"""
  global training_process, model_dir
  print(f"\nReceived signal {signum}. Saving best model before shutdown...")
  if training_process:
    
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

def validate_checkpoint(checkpoint_folder, required_files=None) -> bool:
  """Validate that checkpoint has all required and optional files/directories, returning detailed info"""
  if required_files is None:
    required_files = [TRAINER_PARAMS]

  missing_required_files = []
  missing_required_dirs = []

  for file in required_files:
    path = os.path.join(checkpoint_folder, file)
    if not os.path.exists(path) or not os.path.isfile(path):
      missing_required_files.append(file)

  is_valid = len(missing_required_files) == 0 and len(missing_required_dirs) == 0

  return is_valid

def get_last_valid_checkpoint(model_folder: str):
  """Get the last checkpoint folder that has all required files"""
  assert os.path.exists(model_folder), f"Model folder does not exist: {model_folder}"
  assert os.path.isdir(model_folder), f"Model folder is not a directory: {model_folder}"
  assert os.path.exists(os.path.join(model_folder, "checkpoint")), f"Checkpoint folder does not exist: {os.path.join(model_folder, 'checkpoint')}"

  checkpoint_folders = glob.glob(os.path.join(model_folder, "checkpoint", "checkpoint_*"))
  if not checkpoint_folders:
    return None

  # Sort by modification time, newest first
  checkpoint_folders.sort(key=os.path.getmtime, reverse=True)

  # Find the first valid checkpoint (most recent with all required files)
  for ckpt_folder in checkpoint_folders:
    
    is_valid = validate_checkpoint(ckpt_folder)
    if is_valid:
      return ckpt_folder

  return None

def extract_checkpoint_files(model_dir, target_dir):
  """Extract essential checkpoint files to target directory"""
  essential_files = [BEST_CKPT, TRAINER_PARAMS]

  # Get the last valid checkpoint from the existing model
  last_checkpoint = get_last_valid_checkpoint(model_dir)
  if last_checkpoint is None:
    raise FileNotFoundError(f"No valid checkpoint found in {model_dir}")
  
  extracted_count = 0
  for file in essential_files:
    dst = os.path.join(target_dir, file)
    if os.path.exists(src := os.path.join(last_checkpoint, file)):
      pass
    elif os.path.exists(src := os.path.join(model_dir, file)):
      pass
    else:
      src = None

    if src:
      shutil.copy2(src, dst)
      print(f"Extracted checkpoint file: {file}")
      extracted_count += 1
  
  # Copy best.ckpt directory if it exists
  best_ckpt_src = os.path.join(model_dir, "best.ckpt")
  best_ckpt_dst = os.path.join(target_dir, "best.ckpt")
  if os.path.exists(best_ckpt_src):
    if os.path.exists(best_ckpt_dst):
      shutil.rmtree(best_ckpt_dst)
    shutil.copytree(best_ckpt_src, best_ckpt_dst)
    print("Extracted checkpoint directory: best.ckpt")
    extracted_count += 1
  
  print(f"Extracted {extracted_count} checkpoint files to {target_dir}")
  return extracted_count > 0

def start_initial_training(data_pattern, epochs, output_dir, network):
  """Start training from scratch using train.py logic"""
  global training_process
  if not network:
    network = "cnn=8:3x3,pool=2x2,lstm=32"
  
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


def continue_learning(model_dir, continue_data, network=None, backup=True):
  """Continue learning with proper backup-then-extract sequence using library mode"""
  # Validate model directory
  if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory not found: {model_dir}")
  if not os.path.isdir(model_dir):
    raise ValueError(f"Model path is not a directory: {model_dir}")

  print(f"🔄 Continuing training with additional data (Library Mode)...")
  print(f"📁 Model: {model_dir}")
  print(f"📊 Continue data: {continue_data}")

  if not LIB_MODE:
    print("⚠️ Library mode not available, falling back to CLI mode")
    return continue_learning_cli(model_dir, continue_data, network, backup)

  try:
    # Import ContinueLearningEngine
    from engines.continue_learning import ContinueLearningEngine

    # Create continue learning engine
    engine = ContinueLearningEngine(model_dir)

    # Run continue learning using library mode
    result = engine.continue_learning_sync(
      data_folder=continue_data,
      checkpoint_folder=model_dir,
      network=network,
      backup=backup
    )

    if result["success"]:
      print("✅ Continue learning completed successfully!")
      print(f"New model location: {result.get('model_dir')}")
      if result.get('backup_dir'):
        print(f"Model backup: {result['backup_dir']}")
      print(f"Network: {result.get('network')}")
      print(f"Characters learned: {result.get('chars_count', 0)}")
      return True
    else:
      print(f"❌ Continue learning failed: {result.get('error')}")
      return False

  except KeyboardInterrupt:
    signal_handler(signal.SIGINT, None)
    return False
  except Exception as e:
    print(f"❌ Error during continue learning: {e}")
    return False


def continue_learning_cli(model_dir, continue_data, network=None, backup=True):
  """Continue learning with CLI mode as fallback"""
  # Validate model directory
  if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory not found: {model_dir}")
  if not os.path.isdir(model_dir):
    raise ValueError(f"Model path is not a directory: {model_dir}")

  print(f"🔄 Continuing training with additional data (CLI Mode)...")
  print(f"📁 Model: {model_dir}")
  print(f"📊 Continue data: {continue_data}")

  backup_dir = None
  if backup:
    backup_dir = backup_model(model_dir)
    if backup_dir:
      print(f"📦 Model backed up to: {backup_dir}")
    else:
      print("⚠️  Backup failed, proceeding without backup")

  # Step 2: Create new empty model directory
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"📁 Created new model directory: {model_dir}")

  # Step 3: Extract essential files from backup to new model directory
  if backup_dir and extract_checkpoint_files(backup_dir, model_dir):
    print(f"✅ Extracted essential files from backup to new model directory")
  elif not backup_dir:
    # If no backup was made, extract from the checkpoint directly
    if extract_checkpoint_files(last_checkpoint, model_dir):
      print(f"✅ Extracted essential files to model directory")
    else:
      raise RuntimeError("Failed to extract checkpoint files")
  else:
    raise RuntimeError("Failed to extract checkpoint files from backup")

  # Determine network
  if network is None:
    network = get_current_network(model_dir)
    print(f"Using existing network: {network}")
  else:
    print(f"Using specified network: {network}")

  assert network, "Network is None"

  # Verify dataset
  data_pattern = normalize_data_path(continue_data)
  if not verify_dataset(data_pattern):
    raise ValueError(f"Dataset verification failed for: {data_pattern}")

  # Collect characters for charset extension
  chars = collect_chars(continue_data)
  print(f"Characters in new data: {''.join(chars)} ({len(chars)} total)")

  # Create extended charset file if new characters found
  charset_file = None
  if chars:
    charset_file = os.path.join(model_dir, "extended_charset.txt")
    with open(charset_file, 'w', encoding=UTF_8) as f:
      f.write(''.join(chars))
    print(f"📝 Extended charset file created: {charset_file}")

  # Build training command using extracted files from model directory
  checkpoint_file = os.path.join(model_dir, "best.ckpt")
  if not os.path.exists(checkpoint_file):
    checkpoint_file = os.path.join(model_dir, TRAINER_PARAMS)

  cmd = [
    sys.executable, "-m", "calamari_ocr.scripts.train",
    "--warmstart.model", checkpoint_file,
    "--trainer.auto_upgrade_checkpoints", "True",
    "--trainer.output_dir", model_dir,
    "--train.images", data_pattern,
    "--train.skip_invalid", "True",
    "--train.batch_size", "1",
    "--trainer.gen", "TrainOnly",
    "--early_stopping.n_to_go", "-1",
    "--network", network
  ]

  # Add codec extension if new characters
  if charset_file:
    cmd.extend([
      "--codec.include_files", charset_file,
      "--codec.auto_compute", "True",
      "--codec.keep_loaded", "True"
    ])

  print("🚀 Training command:")
  print(" ".join(f'"{arg}"' if " " in arg else arg for arg in cmd))
  print(f"📁 New model will be created in: {model_dir}")

  global training_process
  try:
    training_process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      encoding=UTF_8,
      errors='replace',
      universal_newlines=True,
      bufsize=1
    )

    # Monitor output in real-time
    for line in iter(training_process.stdout.readline, ''):
      if line.strip():
        cleaned_line = clean_unicode_text(line.strip())
        print(cleaned_line)

    training_process.wait()
    if training_process.returncode == 0:
      print("✅ Training completed successfully!")
      return True
    else:
      print(f"❌ Training failed with code: {training_process.returncode}")
      return False

  except KeyboardInterrupt:
    signal_handler(signal.SIGINT, None)
    return False
  except Exception as e:
    print(f"❌ Error during training: {e}")
    return False
  finally:
    training_process = None

def main():
  # Set up signal handlers for graceful shutdown
  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)
  
  parser = argparse.ArgumentParser(description="Train or continue OCR model")
  parser.add_argument("data_folder", help="Path to training data folder")
  parser.add_argument("model_folder", help="Path to model folder (existing for continuation, new for --new)")
  parser.add_argument("--new", action="store_true", help="Create new model (model_folder must not exist)")
  parser.add_argument("--epochs", type=int, default=5, help="Training epochs for new model")
  parser.add_argument("--network", default=None, help="Network architecture")
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
      success = continue_learning(args.model_folder, data_path, args.network)
      if success:
        print(f"\n🎉 Complete training workflow finished! Model: {args.model_folder}")
      else:
        print("❌ Continuation failed")
        sys.exit(1)
    else:
      print(f"\n🎉 New model created! Model saved to: {args.model_folder}")
  
  # Handle model continuation
  else:
    print(f"🔄 Continuing training on existing model: {args.model_folder}")

    success = continue_learning(args.model_folder, args.data_folder, args.network)
    if success:
      print(f"\n🎉 Model continuation finished! Model: {args.model_folder}")
    else:
      print("❌ Continuation failed")
      sys.exit(1)

if __name__ == "__main__":
  main()

  