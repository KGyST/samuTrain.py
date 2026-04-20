# What is this file for: Backend emulation script that validates learning workflow and database integration using existing src/ functions
# When it is created: 2026-03-28

import argparse
import os
import sqlite3
import sys
from typing import Any, Dict, Optional, Tuple

# Constants (meaningful literals only)
REQUIRED_MODEL_FILES = ["best.ckpt.json", "trainer_params.json"]
DEFAULT_DB_STATS = {
  "total_cases": 0,
  "failset_cases": 0,
  "avg_confidence": 0.0,
  "recent_activity": 0,
  "training_sessions": 0,
}
DEFAULT_PREDICTION_STATS = {"total": 0, "with_predictions": 0, "with_confidence": 0}
BIN_PNG = ".bin.png"

# Add src/ and project root to sys.path
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.dirname(scriptDir)
srcDir = os.path.join(projectRoot, "src")
sys.path.insert(0, srcDir)
sys.path.insert(0, projectRoot)


def _configure_minimal_logging() -> None:
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["PYTHONWARNINGS"] = "ignore"
  os.environ["CALAMARI_LOG_LEVEL"] = "ERROR"
  os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


_configure_minimal_logging()


def _assert_valid_directory(path: str, name: str) -> None:
  assert os.path.isabs(path), f"{name} must be absolute: {path}"
  assert os.path.exists(path), f"{name} not found: {path}"
  assert os.path.isdir(path), f"{name} is not a directory: {path}"


from src.bridge import OCRBridge, normalize_data_path, verify_dataset
from src.db import db


def validate_data_folder(data_folder: str) -> bool:
  _assert_valid_directory(data_folder, "Data folder")
  sNormalized = normalize_data_path(data_folder)
  return verify_dataset(sNormalized)


def validate_model_folder(model_folder: str) -> bool:
  _assert_valid_directory(model_folder, "Model folder")
  for sFile in REQUIRED_MODEL_FILES:
    sFilePath = os.path.join(model_folder, sFile)
    if not os.path.exists(sFilePath):
      return False
  return True


def get_database_stats() -> Dict[str, Any]:
  return db.get_statistics()


def _get_first_bin_png_path(data_folder: str) -> Optional[str]:
  listNames = [f for f in os.listdir(data_folder) if f.endswith(BIN_PNG)]
  if not listNames:
    return None
  return os.path.join(data_folder, listNames[0])


def count_predictions() -> Dict[str, int]:
  with db.get_db_connection() as conn:
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
      """
      SELECT COUNT(*) as count,
             COUNT(CASE WHEN model_prediction IS NOT NULL THEN 1 END) as with_pred,
             COUNT(CASE WHEN confidence > 0 THEN 1 END) as with_conf
      FROM cases
      """
    )
    row = cursor.fetchone()
    if row is None:
      return DEFAULT_PREDICTION_STATS.copy()
    return {
      "total": row["count"],
      "with_predictions": row["with_pred"],
      "with_confidence": row["with_conf"],
    }


def _run_smoke_predict(bridge: Any, data_folder: str) -> bool:
  sTestImage = _get_first_bin_png_path(data_folder)
  if not sTestImage:
    return True
  try:
    bridge.predict(sTestImage)
    return True
  except Exception:
    return False


def _reload_and_predict(bridge: Any, data_folder: str) -> bool:
  if not hasattr(bridge, "learner") or not bridge.learner:
    return True
  try:
    bridge.learner.reload_model()
    sTestImage = _get_first_bin_png_path(data_folder)
    if sTestImage:
      bridge.predict(sTestImage)
    return True
  except Exception:
    return False


def validate_learning_workflow(data_folder: str, model_folder: str) -> bool:
  _assert_valid_directory(data_folder, "Data folder")
  _assert_valid_directory(model_folder, "Model folder")

  if not validate_data_folder(data_folder):
    return False
  if not validate_model_folder(model_folder):
    return False

  os.environ["SAMUTRAIN_DATA_FOLDER"] = data_folder
  os.environ["SAMUTRAIN_MODEL_FOLDER"] = model_folder

  dictStatsInitial = get_database_stats()
  listTrainingCases = db.get_training_cases(include_gt_only=True)
  if not listTrainingCases:
    return False

  try:
    bridge = OCRBridge()
  except Exception:
    return False

  if not _run_smoke_predict(bridge, data_folder):
    return False

  try:
    bridge.continue_learning(
      data_folder=data_folder,
      checkpoint_folder=model_folder,
      network=None,
      backup=True,
      force=False,
    )
  except Exception:
    return False

  dictStatsFinal = get_database_stats()

  iSessionDelta = dictStatsFinal["training_sessions"] - dictStatsInitial["training_sessions"]
  if iSessionDelta <= 0:
    return False

  if not _reload_and_predict(bridge, data_folder):
    return False

  return True


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Backend learning validation script using existing src/ functions",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
  Examples:
  python try_validate_backend_learning.py data/64_case models/generic_ocr_model_3
    """,
  )
  parser.add_argument("data_folder", help="Path to training data folder")
  parser.add_argument("model_folder", help="Path to existing model folder")
  args = parser.parse_args()

  sData = os.path.abspath(args.data_folder)
  sModel = os.path.abspath(args.model_folder)

  bOk = validate_learning_workflow(sData, sModel)

  print("PASS" if bOk else "FAIL")
  sys.exit(0 if bOk else 1)


if __name__ == "__main__":
  main()

