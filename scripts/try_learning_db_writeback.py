#!/usr/bin/env python3
# What is this file for: Minimal learning runner to verify DB writeback after continue_learning
# When it is created: 2026-04-23

import argparse
import json
import os
import sqlite3
import sys
import time
import codecs
from typing import Optional, Tuple

UTF_8 = "utf-8"

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
# os.environ["CALAMARI_LOG_LEVEL"] = "ERROR"
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


def _latest_extended_model_dir(model_folder: str, start_epoch_seconds: int) -> Optional[str]:
  assert os.path.isabs(model_folder), f"model_folder must be absolute: {model_folder}"
  parent_dir = os.path.dirname(model_folder)
  base_name = os.path.basename(model_folder)
  prefix = f"{base_name}_extended_"
  candidates = []
  for entry in os.listdir(parent_dir):
    full_path = os.path.join(parent_dir, entry)
    if os.path.isdir(full_path) and entry.startswith(prefix):
      mtime = int(os.path.getmtime(full_path))
      if mtime >= start_epoch_seconds:
        candidates.append((mtime, full_path))
  if not candidates:
    return None
  candidates.sort(key=lambda item: item[0], reverse=True)
  return candidates[0][1]


def _read_configured_epochs(model_dir: str) -> Optional[int]:
  trainer_params_path = os.path.join(model_dir, "trainer_params.json")
  if not os.path.exists(trainer_params_path):
    return None
  with open(trainer_params_path, "r", encoding=UTF_8) as f:
    params = json.load(f)

  if isinstance(params.get("epochs"), int):
    return int(params["epochs"])
  if isinstance(params.get("params"), dict) and isinstance(params["params"].get("epochs"), int):
    return int(params["params"]["epochs"])
  if isinstance(params.get("scenario"), dict) and isinstance(params["scenario"].get("epochs"), int):
    return int(params["scenario"]["epochs"])
  return None


def run_learning_db_writeback(data_folder: str, model_folder: str, epochs: int = 10) -> Tuple[bool, str]:
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
  print(f"Requested epochs: {epochs}")

  success = bridge.continue_learning(
    data_folder=data_folder,
    checkpoint_folder=model_folder,
    network=None,
    backup=True,
    force=False,
    epochs=epochs,
  )
  if not success:
    return False, "continue_learning returned False"

  after_stats = db.get_statistics()
  after_sessions = int(after_stats.get("training_sessions", 0))
  sessions_delta = after_sessions - before_sessions

  db_path = os.path.abspath(db.db_path)
  updated_cases_count = _count_recent_prediction_updates(db_path, start_epoch_seconds)

  model_out_dir = _latest_extended_model_dir(model_folder, start_epoch_seconds)
  if not model_out_dir:
    return False, "No extended model output directory detected after run"

  configured_epochs = _read_configured_epochs(model_out_dir)
  if configured_epochs is None:
    return False, f"Could not read configured epochs from {model_out_dir}"

  print(f"Persisted trainer epochs: {configured_epochs}")
  if configured_epochs != epochs:
    return False, f"Epoch mismatch (requested={epochs}, persisted={configured_epochs})"

  if sessions_delta <= 0:
    return False, f"No training session increment detected (delta={sessions_delta})"
  if updated_cases_count <= 0:
    return False, f"No DB prediction writeback detected (updated_cases_count={updated_cases_count})"
  return True, "Epoch and DB writeback checks passed"


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Run continue_learning and verify database writeback."
  )
  parser.add_argument("data_folder", help="Path to training data folder")
  parser.add_argument("model_folder", help="Path to existing model folder")
  parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
  args = parser.parse_args()

  data_folder = os.path.abspath(args.data_folder)
  model_folder = os.path.abspath(args.model_folder)
  ok, reason = run_learning_db_writeback(data_folder, model_folder, epochs=args.epochs)
  print(reason)
  print("PASS" if ok else "FAIL")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()

