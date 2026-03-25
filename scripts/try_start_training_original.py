# What is this file for: Backward compatibility wrapper for refactored training system
# When it is created: 2026-03-17 (original), 2026-03-25 (refactored)

import os
import sys
from pathlib import Path

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import the refactored training system
from scripts.try_start_training_refactored import main as refactored_main

# Legacy function stubs for backward compatibility
def normalize_data_path(data_path):
  """Legacy stub - now handled in validation module"""
  from scripts.validation import normalize_data_path as _normalize
  return _normalize(data_path)

def collect_chars(data_folder):
  """Legacy stub - now handled in validation module"""
  from scripts.validation import collect_chars as _collect
  return _collect(data_folder)

def verify_dataset(data_pattern):
  """Legacy stub - now handled in validation module"""
  from scripts.validation import verify_dataset as _verify
  bValid, sError = _verify(data_pattern)
  return bValid

def signal_handler(signum, frame=None):
  """Legacy stub - now handled in refactored main"""
  from scripts.try_start_training_refactored import signal_handler as _handler
  return _handler(signum, frame)

# Legacy main function - delegates to refactored system
def main():
  """Legacy main function - delegates to refactored training system"""
  print("🔄 Using refactored training system...")
  return refactored_main()

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

def signal_handler(signum, teszt = None):
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
    except Exception:
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

def create_base_trainer_params(output_dir, epochs, network):
  """Create base TrainerParams with common settings for both initial and continuation training"""
  from calamari_ocr.ocr.training.params import TrainerParams
  from calamari_ocr.ocr.scenario import CalamariScenario
  
  # Get default trainer params
  trainer_params = CalamariScenario.default_trainer_params()
  
  # Set basic parameters
  trainer_params.output_dir = output_dir
  trainer_params.epochs = epochs
  trainer_params.auto_upgrade_checkpoints = True
  trainer_params.network = network
  
  # Early stopping
  trainer_params.early_stopping.n_to_go = -1  # Disable early stopping
  
  # Training data setup (will be configured by caller)
  trainer_params.gen.train.skip_invalid = True
  trainer_params.gen.setup.train.batch_size = 1
  trainer_params.gen.setup.train.num_processes = 1
  
  # Use training data for validation (TrainOnly) - will be configured by caller
  trainer_params.gen.setup.val.num_processes = 1
  
  # Codec settings
  trainer_params.codec.auto_compute = True
  trainer_params.codec.keep_loaded = True
  
  # Disable progress bar for cleaner output
  trainer_params.progress_bar = False
  
  return trainer_params

def create_trainer_params_for_initial_training(data_pattern, epochs, output_dir, network):
  """Create TrainerParams for initial training from scratch using library API"""
  trainer_params = create_base_trainer_params(
    output_dir, epochs, network if network else "cnn=8:3x3,pool=2x2,lstm=32"
  )
  
  # Set training data for initial training
  trainer_params.gen.train.images = [data_pattern]
  trainer_params.gen.val.images = trainer_params.gen.train.images
  
  # No warmstart for initial training
  trainer_params.warmstart.model = None
  
  return trainer_params

def setup_charset_extension(trainer_params, chars, model_dir):
  """Setup charset extension if new characters are found"""
  charset_file = None
  if chars:
    charset_file = os.path.join(model_dir, "extended_charset.txt")
    with open(charset_file, 'w', encoding=UTF_8) as f:
      f.write(''.join(chars))
    
    trainer_params.codec.include_files = [charset_file]
    trainer_params.codec.auto_compute = True
    trainer_params.codec.keep_loaded = True
    print(f"📝 Extended charset file created: {charset_file}")
  
  return charset_file

def setup_library_training_environment():
  """Setup common environment for library-based training"""
  # Force minimal logging
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["PYTHONWARNINGS"] = "ignore"
  os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'
  
  # Protocol Buffers compatibility
  os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def start_initial_training(data_pattern, epochs, output_dir, network):
  """Start training from scratch using library API (instead of CLI)"""
  global training_process
  
  print("🚀 Starting initial training from scratch (Library Mode)...")
  print(f"📊 Data: {data_pattern}")
  print(f"🎯 Epochs: {epochs}")
  print(f"📁 Output: {output_dir}")
  
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  # Convert to absolute path
  data_pattern = os.path.abspath(data_pattern)
  output_dir = os.path.abspath(output_dir)
  
  # Setup environment
  setup_library_training_environment()
  
  # Verify dataset
  if not verify_dataset(data_pattern):
    raise ValueError(f"Dataset verification failed for: {data_pattern}")
  
  try:
    # Create trainer parameters using library API
    trainer_params = create_trainer_params_for_initial_training(
      data_pattern, epochs, output_dir, network
    )
    
    print(f"🔧 Using network: {trainer_params.network}")
    print(f"📁 Model will be saved to: {output_dir}")
    
    # Run training using library API (same as continue_learning)
    result = calamari_train(trainer_params)
    
    if result is None or (hasattr(result, 'returncode') and result.returncode != 0):
      print("❌ Initial training failed")
      return False
    
    print("✅ Initial training completed successfully!")
    return True
    
  except KeyboardInterrupt:
    signal_handler(signal.SIGINT)
    return False
  except Exception as e:
    print(f"❌ Error during initial training: {e}")
    return False

def continue_learning(model_dir, continue_data, network=None, backup=True):
  """Continue learning using library API with backup and file handling"""
  # Validate model directory
  if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory not found: {model_dir}")
  if not os.path.isdir(model_dir):
    raise ValueError(f"Model path is not a directory: {model_dir}")

  print(f"🔄 Continuing training with additional data (Library Mode)...")
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
    last_checkpoint = get_last_valid_checkpoint(model_dir)
    if last_checkpoint and extract_checkpoint_files(last_checkpoint, model_dir):
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

  try:
    # Create base trainer parameters using shared function
    trainer_params = create_base_trainer_params(model_dir, 100, network)
    
    # Set training data for continuation learning
    trainer_params.gen.train.images = [data_pattern]
    trainer_params.gen.val.images = trainer_params.gen.train.images
    
    # Warmstart parameters
    checkpoint_file = os.path.join(model_dir, "best.ckpt")
    if not os.path.exists(checkpoint_file):
      checkpoint_file = os.path.join(model_dir, TRAINER_PARAMS)
    
    if not os.path.exists(checkpoint_file):
      raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    trainer_params.warmstart.model = checkpoint_file
    trainer_params.warmstart.allow_partial = True
    trainer_params.warmstart.trim_graph_name = False
    
    # Setup charset extension using shared function
    charset_file = setup_charset_extension(trainer_params, chars, model_dir)

    print("🚀 Starting training with library API...")
    print(f"📁 New model will be created in: {model_dir}")
    print(f"🔧 Using checkpoint: {checkpoint_file}")
    print(f"🌐 Network: {network}")

    # Run training using library API
    result = calamari_train(trainer_params)
    
    print("✅ Training completed successfully!")
    return True
  except KeyboardInterrupt:
    signal_handler(signal.SIGINT)
    return False
  except Exception as e:
    print(f"❌ Error during training: {e}")
    return False

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

  