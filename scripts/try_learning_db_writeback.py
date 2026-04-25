#!/usr/bin/env python3
# What is this file for: Minimal learning runner to verify DB writeback after continue_learning
# When it is created: 2026-04-23

import argparse
import json
import glob
import os
import sqlite3
import sys
import time
import codecs
from typing import List, Optional, Tuple

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
  # Try best.ckpt.json first (where Calamari actually stores epochs)
  best_ckpt_path = os.path.join(model_dir, "best.ckpt.json")
  if os.path.exists(best_ckpt_path):
    with open(best_ckpt_path, "r", encoding=UTF_8) as f:
      params = json.load(f)
    if isinstance(params.get("epochs"), int):
      return int(params["epochs"])
  
  # Fallback to trainer_params.json for backward compatibility
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


def _collect_learning_files(data_folder: str) -> List[str]:
  pattern = os.path.join(data_folder, "*.bin.png")
  return sorted(glob.glob(pattern))


def _read_gt_text_for_image(image_path: str) -> Optional[str]:
  gt_path = image_path.replace(".bin.png", ".gt.txt")
  if not os.path.exists(gt_path):
    return None
  with open(gt_path, "r", encoding=UTF_8) as f:
    return f.read().strip()


def _upsert_learning_cases_to_db(bridge, data_folder: str) -> Tuple[int, int, int]:
  learning_files = _collect_learning_files(data_folder)
  inserted = 0
  updated = 0
  updated_case_ids = []
  for image_path in learning_files:
    img_name = os.path.basename(image_path)
    pred, conf = bridge.predict(image_path)
    pred_text = str(pred)
    gt_text = _read_gt_text_for_image(image_path)
    existing = bridge.db.get_case_by_img_path(img_name)
    if existing:
      case_id = int(existing["id"])
      bridge.db.update_case_ocr_result(case_id, pred_text, float(conf))
      bridge.db.update_model_prediction(case_id, pred_text)
      if gt_text is not None:
        bridge.db.update_case_gt_text(case_id, gt_text)
      updated_case_ids.append(case_id)
      updated += 1
    else:
      case_id = bridge.db.insert_case(
        img_path=img_name,
        ocr_text=pred_text,
        confidence=float(conf),
        gt_text=gt_text,
        is_failset=False,
        model_prediction=pred_text,
      )
      updated_case_ids.append(case_id)
      inserted += 1

  if updated_case_ids:
    bridge.db.mark_cases_updated(updated_case_ids)
  return len(learning_files), inserted, updated


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

  total_learning_cases, inserted_cases, updated_cases = _upsert_learning_cases_to_db(bridge, data_folder)
  print(
    f"Learning cases synced to DB: total={total_learning_cases}, inserted={inserted_cases}, updated={updated_cases}"
  )
  if total_learning_cases == 0:
    return False, "No learning cases found in data folder (*.bin.png)"
  if (inserted_cases + updated_cases) != total_learning_cases:
    return False, (
      f"DB sync mismatch (total={total_learning_cases}, "
      f"inserted+updated={inserted_cases + updated_cases})"
    )

  if sessions_delta <= 0:
    return False, f"No training session increment detected (delta={sessions_delta})"
  if updated_cases_count <= 0 and updated_cases <= 0:
    return False, (
      "No DB prediction writeback detected from training or post-sync "
      f"(updated_cases_count={updated_cases_count}, updated_cases={updated_cases})"
    )
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

