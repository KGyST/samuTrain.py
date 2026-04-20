# What is this file for: Test script for try_validate_backend_learning
# When it is created: 2026-03-31

import os
import sys

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)

from try_validate_backend_learning import validate_backend_learning

def test_validate_backend_learning():
  """Test the validate_backend_learning function with different parameters"""
  
  # Test case 1: Basic validation with single_case data
  validate_backend_learning(
    os.path.join(project_root, "data/single_case"), 
    os.path.join(project_root, "models/generic_ocr_model")
  )

if __name__ == "__main__":
  test_validate_backend_learning()
