# What is this file for: Backend emulation script that validates learning workflow and database integration using existing src/ functions
# When it is created: 2026-03-28

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple

# Constants (meaningful literals only)
REQUIRED_MODEL_FILES = ["best.ckpt.json", "trainer_params.json"]
DEFAULT_DB_STATS = {
    'total_cases': 0,
    'failset_cases': 0,
    'avg_confidence': 0.0,
    'recent_activity': 0,
    'training_sessions': 0
}
DEFAULT_PREDICTION_STATS = {'total': 0, 'with_predictions': 0, 'with_confidence': 0}
UTF_8 = 'utf-8'
BIN_PNG = ".bin.png"
GT_TXT = ".gt.txt"
IMPORT_ERROR_MSG = "Import error: {}"

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Set environment for minimal logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Set UTF-8 encoding for stdout to handle emoji characters
if sys.platform == "win32":
  import codecs
  sys.stdout = codecs.getwriter(UTF_8)(sys.stdout.detach())
  sys.stderr = codecs.getwriter(UTF_8)(sys.stderr.detach())

def _assert_valid_directory(path: str, name: str) -> None:
  """Helper function to validate directory path"""
  assert os.path.isabs(path), f"{name} must be absolute: {path}"
  assert os.path.exists(path), f"{name} not found: {path}"
  assert os.path.isdir(path), f"{name} is not a directory: {path}"

def validate_data_folder(data_folder: str) -> bool:
  """Validate that data folder exists and contains training data using existing functions"""
  _assert_valid_directory(data_folder, "Data folder")
  
  try:
    from src.bridge import verify_dataset, normalize_data_path
    
    normalizedPath = normalize_data_path(data_folder)
    return verify_dataset(normalizedPath)
      
  except ImportError:
    return False

def validate_model_folder(model_folder: str) -> bool:
  """Validate that model folder exists and contains required files"""
  _assert_valid_directory(model_folder, "Model folder")
  
  for file in REQUIRED_MODEL_FILES:
    file_path = os.path.join(model_folder, file)
    if not os.path.exists(file_path):
      return False
  
  return True

def get_database_stats() -> Dict[str, Any]:
  """Get current database statistics using existing db function"""
  try:
    from src.db import db
    return db.get_statistics()
  except ImportError as e:
    return DEFAULT_DB_STATS.copy()

def _get_test_image(data_folder: str) -> str:
  """Helper function to get a test image for prediction"""
  test_images = [f for f in os.listdir(data_folder) if f.endswith(BIN_PNG)]
  if test_images:
    return os.path.join(data_folder, test_images[0])
  return None

def count_predictions() -> Dict[str, int]:
  """Count cases with predictions in database using existing db function"""
  try:
    from src.db import db
    with db.get_db_connection() as conn:
      cursor = conn.execute("""
        SELECT COUNT(*) as count,
               COUNT(CASE WHEN model_prediction IS NOT NULL THEN 1 END) as with_pred,
               COUNT(CASE WHEN confidence > 0 THEN 1 END) as with_conf
        FROM cases
      """)
      result = cursor.fetchone()
      return {
        'total': result['count'],
        'with_predictions': result['with_pred'],
        'with_confidence': result['with_conf']
      }
  except Exception:
    return DEFAULT_PREDICTION_STATS.copy()


def validate_learning_workflow(data_folder: str, model_folder: str) -> bool:
  """Main validation function for learning workflow using existing src functions"""
  _assert_valid_directory(data_folder, "Data folder")
  _assert_valid_directory(model_folder, "Model folder")
  
  try:
    from src.bridge import OCRBridge
    from src.db import db
  except ImportError:
    return False
  
  dataValid = validate_data_folder(data_folder)
  if not dataValid:
    return False
  
  modelValid = validate_model_folder(model_folder)
  if not modelValid:
    return False
  
  os.environ['SAMUTRAIN_DATA_FOLDER'] = data_folder
  os.environ['SAMUTRAIN_MODEL_FOLDER'] = model_folder
  
  initialStats = get_database_stats()
  initialPredictions = count_predictions()
  
  training_cases = db.get_training_cases(include_gt_only=True)
  
  if not training_cases:
    return False
  
  try:
    bridge = OCRBridge()
    
    test_img = _get_test_image(data_folder)
    if test_img:
      pred, conf = bridge.predict(test_img)
      
  except Exception:
    return False
  
  try:
    success = bridge.continue_learning(
      data_folder=data_folder,
      checkpoint_folder=model_folder,
      network=None,
      backup=True,
      force=False
    )
    
    if not success:
      return False
      
  except Exception:
    return False
  
  finalStats = get_database_stats()
  finalPredictions = count_predictions()
  
  predChange = finalPredictions['with_predictions'] - initialPredictions['with_predictions']
  confChange = finalPredictions['with_confidence'] - initialPredictions['with_confidence']
  sessionChange = finalStats['training_sessions'] - initialStats['training_sessions']
  
  validationPassed = True
  
  if sessionChange <= 0:
    validationPassed = False
  
  try:
    if hasattr(bridge, 'learner') and bridge.learner:
      bridge.learner.reload_model()
      
      test_img = _get_test_image(data_folder)
      if test_img:
        predAfter, confAfter = bridge.predict(test_img)
    
  except Exception:
    validationPassed = False
  
  try:
    from src.db import get_db_connection
    with get_db_connection() as conn:
      cursor = conn.execute("""
        SELECT COUNT(*) as count,
               COUNT(CASE WHEN model_prediction != ocr_text THEN 1 END) as different_pred,
               COUNT(CASE WHEN model_prediction IS NOT NULL AND ocr_text IS NOT NULL THEN 1 END) as both_populated
        FROM cases
      """)
      dbResult = cursor.fetchone()
        
  except Exception:
    pass
  
  return validationPassed

def main() -> None:
  parser = argparse.ArgumentParser(
    description="Backend learning validation script using existing src/ functions",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python try_validate_backend_learning.py data/64_case models/generic_ocr_model_3
  python try_validate_backend_learning.py --force data/new_case models/existing_model
        """
  )
  
  parser.add_argument("data_folder", help="Path to training data folder")
  parser.add_argument("model_folder", help="Path to existing model folder")
  parser.add_argument("--force", action="store_true", help="Force validation even if checks fail")
  
  args = parser.parse_args()
  
  dataFolder = os.path.abspath(args.data_folder)
  modelFolder = os.path.abspath(args.model_folder)
  
  success = validate_learning_workflow(dataFolder, modelFolder)
  
  if success:
    sys.exit(0)
  else:
    if args.force:
      sys.exit(0)
    else:
      sys.exit(1)

if __name__ == "__main__":
  main()
