# What is this file for: Script to resume learning from a checkpoint, run one additional epoch, and produce final model with validation.
# When it is created: 2026-03-04

import os
import sys
import subprocess

# Force minimal logging at the OS level (3 = ERROR only)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

def get_current_epoch(model_folder):
  try:
    ckpt_dir = os.path.join(model_folder, "checkpoint")
    if not os.path.exists(ckpt_dir): return 1
    ckpts = [int(d.split("_")[1]) for d in os.listdir(ckpt_dir) if d.startswith("checkpoint_")]
    return max(ckpts) if ckpts else 1
  except: return 1

def finish_epoch(data_folder, checkpoint_folder):
  import json
  params_path = os.path.join(checkpoint_folder, "trainer_params.json")
  with open(params_path, 'r') as f:
    params = json.load(f)
  params["gen"]["train"]["images"] = [os.path.join(data_folder, "*.bin.png")]
  current = get_current_epoch(checkpoint_folder)
  params["epochs"] = current + 1
  params["val_every_n"] = 1
  with open(params_path, 'w') as f:
    json.dump(params, f, indent=2)
  cmd = [
    sys.executable, "-m", "calamari_ocr.scripts.resume_training",
    checkpoint_folder
  ]
  subprocess.run(cmd)

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python try_finish_epoch.py <data_folder> <checkpoint_folder>")
    sys.exit(1)
  data_folder = sys.argv[1]
  checkpoint_folder = sys.argv[2]
  finish_epoch(data_folder, checkpoint_folder)
