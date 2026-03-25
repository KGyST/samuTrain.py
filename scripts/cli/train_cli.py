# What is this file for: CLI interface for training operations in samuTrain
# When it is created: 2026-03-25

import argparse
import os
import sys
from typing import Tuple, Optional
from pathlib import Path

# Add project root to sys.path
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.dirname(os.path.dirname(scriptDir))
sys.path.insert(0, projectRoot)

from scripts.validation import validate_training_args, normalize_data_path
from scripts.constants import DEFAULT_EPOCHS, DEFAULT_NETWORK


def create_argument_parser() -> argparse.ArgumentParser:
  """Create and configure argument parser for training CLI"""
  
  parser = argparse.ArgumentParser(
    description="Train or continue OCR model using samuTrain architecture",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Create new model
  python try_start_training.py data/64_case models/new_model --new --epochs 10
  
  # Continue existing model
  python try_start_training.py data/new_case models/existing_model
  
  # Auto-continue after initial training
  python try_start_training.py data/64_case models/auto_model --new --auto-continue
    """
  )
  
  # Positional arguments
  parser.add_argument(
    "data_folder",
    help="Path to training data folder containing .bin.png and .gt.txt files"
  )
  
  parser.add_argument(
    "model_folder", 
    help="Path to model folder (existing for continuation, new for --new)"
  )
  
  # Optional flags
  parser.add_argument(
    "--new",
    action="store_true",
    help="Create new model (model_folder must not exist)"
  )
  
  parser.add_argument(
    "--epochs",
    type=int,
    default=DEFAULT_EPOCHS,
    help=f"Training epochs for new model (default: {DEFAULT_EPOCHS})"
  )
  
  parser.add_argument(
    "--network",
    default=None,
    help=f"Network architecture (default: {DEFAULT_NETWORK})"
  )
  
  parser.add_argument(
    "--auto-continue",
    action="store_true", 
    help="Automatically continue with same data after initial training"
  )
  
  parser.add_argument(
    "--force",
    action="store_true",
    help="Force overwrite existing model directory without prompt"
  )
  
  return parser


def parse_and_validate_args(args: Optional[list] = None) -> Tuple[argparse.Namespace, bool]:
  """Parse command line arguments and perform validation"""
  
  parser = create_argument_parser()
  parsedArgs = parser.parse_args(args)
  
  # Validate arguments
  bValid, sError = validate_training_args(parsedArgs)
  
  if not bValid:
    print(f"❌ Validation Error: {sError}")
    return parsedArgs, False
  
  # Normalize data path
  parsedArgs.data_folder = normalize_data_path(parsedArgs.data_folder)
  
  return parsedArgs, True


def main():
  """Main CLI entry point"""
  
  try:
    args, bValid = parse_and_validate_args()
    
    if not bValid:
      sys.exit(1)
    
    return args
    
  except KeyboardInterrupt:
    print("\n⚠️ Operation cancelled by user")
    sys.exit(1)
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
