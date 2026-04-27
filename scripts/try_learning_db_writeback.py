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
import logging
import re
import threading
from typing import List, Optional, Tuple

UTF_8 = "utf-8"

class EvaluationLogHandler(logging.Handler):
    """Custom logging handler to capture Calamari evaluation output during training"""
    
    def __init__(self, db, session_id: str, data_files: List[str]):
        super().__init__()
        self.db = db
        self.session_id = session_id
        self.data_files = data_files
        self.current_epoch = 0
        self.evaluation_count = 0
        self.buffer_lock = threading.Lock()
        self.file_index = 0
        
    def set_epoch(self, epoch: int):
        """Set current training epoch"""
        self.current_epoch = epoch
        self.file_index = 0  # Reset file index for new epoch
        
    def parse_evaluation_log(self, log_line: str) -> Optional[dict]:
        """Parse evaluation log line: 'CER: 1.0\n  PRED: '1'\n  TRUE: 'Zebegényi SZOT Üdülő''"""
        # Handle multi-line evaluation logs
        lines = log_line.strip().split('\n')
        if len(lines) < 3:
            return None
            
        cer_match = None
        pred_match = None
        true_match = None
        
        for line in lines:
            if 'CER:' in line:
                cer_match = re.search(r'CER:\s*([\d.]+)', line)
            elif 'PRED:' in line:
                pred_match = re.search(r"PRED:\s*'([^']*)'", line)
            elif 'TRUE:' in line:
                true_match = re.search(r"TRUE:\s*'([^']*)'", line)
        
        if cer_match and pred_match and true_match:
            return {
                'cer': float(cer_match.group(1)),
                'prediction': pred_match.group(1),
                'ground_truth': true_match.group(1)
            }
        return None
    
    def get_current_image_file(self) -> Optional[str]:
        """Get current image file being evaluated based on order"""
        if self.file_index < len(self.data_files):
            img_file = self.data_files[self.file_index]
            self.file_index += 1
            return img_file
        return None
    
    def emit(self, record):
        """Handle log record emission"""
        try:
            msg = self.format(record)
            
            # Check if this is an evaluation log
            if "CER:" in msg and "PRED:" in msg and "TRUE:" in msg:
                evaluation_data = self.parse_evaluation_log(msg)
                
                if evaluation_data:
                    img_file = self.get_current_image_file()
                    if img_file:
                        # Insert into database immediately
                        self.db.insert_training_evaluation(
                            self.session_id,
                            self.current_epoch,
                            img_file,
                            evaluation_data['cer'],
                            evaluation_data['prediction'],
                            evaluation_data['ground_truth']
                        )
                        
                        with self.buffer_lock:
                            self.evaluation_count += 1
                        
        except Exception as e:
            # Don't let logging errors break training
            print(f"⚠️ Evaluation logging error: {e}")
    
    def get_evaluation_count(self) -> int:
        """Get total evaluation count"""
        with self.buffer_lock:
            return self.evaluation_count

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

  # Setup evaluation logging to capture data during training
  data_files = [os.path.basename(f) for f in _collect_learning_files(data_folder)]
  session_id = f"eval_session_{int(start_epoch_seconds)}"
  evaluation_handler = EvaluationLogHandler(db, session_id, data_files)
  
  # Get the tfaip model logger (this is where Calamari outputs evaluation data)
  tfaip_logger = logging.getLogger('tfaip.model.print_evaluate_lay')
  tfaip_logger.addHandler(evaluation_handler)
  tfaip_logger.setLevel(logging.INFO)
  
  print(f"🔧 Evaluation logging setup for {len(data_files)} files in session {session_id[:8]}")

  success = bridge.continue_learning(
    data_folder=data_folder,
    checkpoint_folder=model_folder,
    network=None,
    backup=True,
    force=False,
    epochs=epochs,
  )
  
  # Cleanup evaluation logging
  tfaip_logger.removeHandler(evaluation_handler)
  evaluation_count = evaluation_handler.get_evaluation_count()
  
  if not success:
    return False, "continue_learning returned False"

  after_stats = db.get_statistics()
  after_sessions = int(after_stats.get("training_sessions", 0))
  sessions_delta = after_sessions - before_sessions

  model_out_dir = _latest_extended_model_dir(model_folder, start_epoch_seconds)
  if not model_out_dir:
    return False, "No extended model output directory detected after run"

  configured_epochs = _read_configured_epochs(model_out_dir)
  if configured_epochs is None:
    return False, f"Could not read configured epochs from {model_out_dir}"

  print(f"Persisted trainer epochs: {configured_epochs}")
  if configured_epochs != epochs:
    return False, f"Epoch mismatch (requested={epochs}, persisted={configured_epochs})"

  # Verify evaluation data was captured during training
  print(f"Evaluations captured during training: {evaluation_count}")
  
  # Get evaluation statistics from database
  try:
    evaluation_records = db.get_training_evaluations(session_id)
    print(f"Evaluation records in database: {len(evaluation_records)}")
    
    if len(evaluation_records) > 0:
      # Show some sample evaluation data
      print("\nSample evaluation data:")
      for i, record in enumerate(evaluation_records[:3]):
        print(f"  {i+1}. CER: {record['cer']:.3f}, PRED: '{record['prediction']}', TRUE: '{record['ground_truth']}'")
      
      if len(evaluation_records) > 3:
        print(f"  ... and {len(evaluation_records) - 3} more")
    else:
      print("⚠️ No evaluation records found in database")
      
  except Exception as e:
    print(f"⚠️ Error reading evaluation data: {e}")

  if sessions_delta <= 0:
    return False, f"No training session increment detected (delta={sessions_delta})"
  if evaluation_count == 0:
    return False, "No evaluation data captured during training"
    
  return True, f"Training completed with {evaluation_count} evaluations captured"


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Run continue_learning and capture evaluation data during training."
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

