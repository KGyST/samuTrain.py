# What is this file for: Input validation helpers for samuTrain scripts
# When it is created: 2026-03-25

import os
import glob
from typing import Tuple, Optional
from pathlib import Path


def assert_directory_exists(sPath: str, sName: str = "Directory"):
  """Helper function to assert directory exists"""
  assert os.path.exists(sPath), f"{sName} not found: {sPath}"
  assert os.path.isdir(sPath), f"{sName} is not a directory: {sPath}"


def assert_file_exists(sPath: str, sName: str = "File"):
  """Helper function to assert file exists"""
  assert os.path.exists(sPath), f"{sName} not found: {sPath}"
  assert os.path.isfile(sPath), f"{sName} is not a file: {sPath}"


def normalize_data_path(data_path: str) -> str:
  """Normalize data path to include glob pattern if needed"""
  
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


def verify_dataset(data_pattern: str) -> Tuple[bool, Optional[str]]:
  """Verify that dataset exists and has corresponding ground truth files"""
  
  images = glob.glob(data_pattern)
  if not images:
    return False, f"No images found matching: {data_pattern}"
  
  for img in images:
    gt_file = img.replace(".bin.png", ".gt.txt")
    if not os.path.exists(gt_file):
      return False, f"Missing Ground Truth file: {gt_file}"
  
  return True, None


def validate_training_args(args) -> Tuple[bool, Optional[str]]:
  """Validate training arguments and return (is_valid, error_message)"""
  
  # Validate data folder (before normalization)
  if not os.path.exists(args.data_folder):
    return False, f"Data folder not found: {args.data_folder}"
  
  if not os.path.isdir(args.data_folder):
    return False, f"Data path is not a directory: {args.data_folder}"
  
  # Verify dataset (after normalization)
  data_pattern = normalize_data_path(args.data_folder)
  bValid, sError = verify_dataset(data_pattern)
  if not bValid:
    return False, sError
  
  # Check model folder existence
  model_exists = os.path.exists(args.model_folder)
  
  # Validate arguments based on operation type
  if args.new and model_exists:
    if not args.force:
      return False, f"Model folder already exists: {args.model_folder}. Use --force to overwrite or choose a different folder"
  elif not args.new and not model_exists:
    return False, f"Model folder does not exist: {args.model_folder}. Use --new to create a new model or provide an existing model folder"
  
  # Validate epochs
  if args.epochs <= 0:
    return False, f"Epochs must be positive, got: {args.epochs}"
  
  # Validate network format (basic check)
  if args.network and "=" not in args.network:
    return False, f"Invalid network format: {args.network}. Expected format like 'cnn=8:3x3,pool=2x2,lstm=32'"
  
  return True, None


def validate_model_directory(model_dir: str) -> Tuple[bool, Optional[str]]:
  """Validate that model directory has required files for continuation"""
  
  assert_directory_exists(model_dir, "Model directory")
  
  # Check for trainer_params.json
  trainer_params_path = os.path.join(model_dir, "trainer_params.json")
  if not os.path.exists(trainer_params_path):
    return False, f"Missing trainer_params.json in model directory: {model_dir}"
  
  # Check for checkpoint directory
  checkpoint_dir = os.path.join(model_dir, "checkpoint")
  if not os.path.exists(checkpoint_dir):
    return False, f"Missing checkpoint directory in: {model_dir}"
  
  # Check for best.ckpt or best.ckpt.json
  best_ckpt_path = os.path.join(model_dir, "best.ckpt")
  best_ckpt_json = os.path.join(model_dir, "best.ckpt.json")
  
  if not os.path.exists(best_ckpt_path) and not os.path.exists(best_ckpt_json):
    return False, f"Missing best.ckpt or best.ckpt.json in model directory: {model_dir}"
  
  return True, None


def collect_chars(data_folder: str) -> list:
  """Collect unique characters from all .gt.txt files in data folder"""
  
  chars = set()
  data_path = Path(data_folder)
  
  assert_directory_exists(data_folder, "Data folder")
  
  for gt_file in data_path.glob("*.gt.txt"):
    try:
      with open(gt_file, 'r', encoding='utf-8') as f:
        content = f.read()
        chars.update(content)
    except Exception as e:
      print(f"⚠️ Warning: Failed to read {gt_file}: {e}")
  
  return sorted(list(chars))
