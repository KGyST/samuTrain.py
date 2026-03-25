# What is this file for: Clean version of refactored training script
# When it is created: 2026-03-25

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
  """Legacy stub - now handled by TrainingInterruptionManager"""
  from scripts.try_start_training_refactored import interruption_manager
  interruption_manager.request_shutdown(f"Signal {signum} received")
  return interruption_manager._signal_handler(signum, frame)

# Legacy main function - delegates to refactored system
def main():
  """Legacy main function - delegates to refactored training system"""
  print("🔄 Using refactored training system...")
  return refactored_main()

if __name__ == "__main__":
  main()
