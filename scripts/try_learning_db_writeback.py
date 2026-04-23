#!/usr/bin/env python3
# What is this file for: Minimal learning runner to verify DB writeback after continue_learning
# When it is created: 2026-04-23

import argparse
import os
import sqlite3
import sys
import time
import codecs

UTF_8 = "utf-8"

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["CALAMARI_LOG_LEVEL"] = "ERROR"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
if sys.platform == "win32":
  sys.stdout = codecs.getwriter(UTF_8)(sys.stdout.detach())


def _count_recent_prediction_updates(db_path: str, start_epoch_seconds: int) -> int:
  assert os.path.isabs(db_path), f"db_path must be absolute: {db_path}"
  assert os.path.exists(db_path), f"Database file not found: {db_path}"
  with sqlite3.connect(db_path) as conn:
    cursor = conn.execute(
      """
      SELECT COUNT(*)
      FROM cases
      WHERE updated_flag = TRUE
        AND last_updated >= datetime(?, 'unixepoch')
      """,
      (start_epoch_seconds,),
    )
    return int(cursor.fetchone()[0])


def run_learning_db_writeback(data_folder: str, model_folder: str, epochs: int = 1) -> bool:
  assert os.path.isabs(data_folder), f"data_folder must be absolute: {data_folder}"
  assert os.path.isabs(model_folder), f"model_folder must be absolute: {model_folder}"
  assert os.path.isdir(data_folder), f"Data folder not found: {data_folder}"
  assert os.path.isdir(model_folder), f"Model folder not found: {model_folder}"
  assert epochs > 0, f"epochs must be positive, got {epochs}"

  os.environ["SAMUTRAIN_DATA_FOLDER"] = data_folder
  os.environ["SAMUTRAIN_MODEL_FOLDER"] = model_folder

  from src.db import db
  from src.bridge import OCRBridge

  bridge = OCRBridge(db)
  before_stats = db.get_statistics()
  before_sessions = int(before_stats.get("training_sessions", 0))
  start_epoch_seconds = int(time.time())

  success = bridge.continue_learning(
    data_folder=data_folder,
    checkpoint_folder=model_folder,
    network=None,
    backup=True,
    force=False,
    epochs=epochs,
  )
  if not success:
    return False

  after_stats = db.get_statistics()
  after_sessions = int(after_stats.get("training_sessions", 0))
  sessions_delta = after_sessions - before_sessions

  db_path = os.path.abspath(db.db_path)
  updated_cases_count = _count_recent_prediction_updates(db_path, start_epoch_seconds)
  return sessions_delta > 0 and updated_cases_count > 0


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Run continue_learning and verify database writeback."
  )
  parser.add_argument("data_folder", help="Path to training data folder")
  parser.add_argument("model_folder", help="Path to existing model folder")
  parser.add_argument("--epochs", type=int, default=1, help="Training epochs (default: 1)")
  args = parser.parse_args()

  data_folder = os.path.abspath(args.data_folder)
  model_folder = os.path.abspath(args.model_folder)
  ok = run_learning_db_writeback(data_folder, model_folder, epochs=args.epochs)
  print("PASS" if ok else "FAIL")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()

