# What is this file for: Script to continue learning from a checkpoint using given data folder.
# When it is created: 2026-03-04

import os
import sys
import subprocess
import json
import glob

# Force minimal logging at the OS level (3 = ERROR only)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

def collect_chars(data_folder):
  chars = set()
  for root, dirs, files in os.walk(data_folder):
    for file in files:
      if file.endswith('.gt.txt'):
        gt_file = os.path.join(root, file)
        try:
          with open(gt_file, 'r', encoding='utf-8') as f:
            chars.update(f.read())
        except:
          pass
  return sorted(list(chars))

def continue_learning(data_folder, checkpoint_folder):
  import json
  params_path = os.path.join(checkpoint_folder, "trainer_params.json")
  with open(params_path, 'r') as f:
    params = json.load(f)
  params["gen"]["train"]["images"] = [os.path.join(data_folder, "*.bin.png")]
  params["skip_invalid_gt"] = True
  
  # Collect characters and update only the codec construction params
  chars = collect_chars(data_folder)
  if chars:
    # Only update codec construction params, don't modify the scenario codec
    # This allows Calamari's codec extension mechanism to work properly
    params["codec"]["include"] = list(chars)
    params["codec"]["auto_compute"] = True
    params["codec"]["keep_loaded"] = True  # This enables codec extension
  
  # Disable preloading to avoid codec KeyError during data loading
  params["gen"]["train"]["preload"] = False
  if "setup" in params["gen"] and "train" in params["gen"]["setup"]:
    params["gen"]["setup"]["train"]["preload"] = False
  
  with open(params_path, 'w') as f:
    json.dump(params, f, indent=2)
  cmd = [
    sys.executable, "-m", "calamari_ocr.scripts.resume_training",
    checkpoint_folder
  ]
  subprocess.run(cmd)

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python try_continue_learning.py <data_folder> <checkpoint_folder>")
    sys.exit(1)
  data_folder = sys.argv[1]
  checkpoint_folder = sys.argv[2]
  continue_learning(data_folder, checkpoint_folder)
