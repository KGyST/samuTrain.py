# What is this file for: Debug codec construction to understand why new characters aren't being added
# When it is created: 2026-03-09

import json
import os
import sys

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

def debug_codec(data_folder, checkpoint_folder):
  # Load current params
  params_path = os.path.join(checkpoint_folder, "trainer_params.json")
  with open(params_path, 'r') as f:
    params = json.load(f)
  
  # Collect characters from GT files
  chars = collect_chars(data_folder)
  print("Characters from GT files:", chars)
  
  # Show current codec settings
  print("Current codec include:", params["codec"].get("include", []))
  print("Current resolved_include_chars:", params["codec"].get("resolved_include_chars", []))
  print("auto_compute:", params["codec"].get("auto_compute"))
  print("keep_loaded:", params["codec"].get("keep_loaded"))
  
  # Show model's current codec
  model_path = os.path.join(checkpoint_folder, "best.ckpt.json")
  if os.path.exists(model_path):
    with open(model_path, 'r') as f:
      model_params = json.load(f)
    model_charset = model_params["scenario"]["data"]["codec"]["charset"]
    print("Model's current charset:", model_charset)
    
    # Check which chars are missing
    missing_chars = [c for c in chars if c not in model_charset]
    print("Missing characters:", missing_chars)

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python try_debug_codec.py <data_folder> <checkpoint_folder>")
    sys.exit(1)
  data_folder = sys.argv[1]
  checkpoint_folder = sys.argv[2]
  debug_codec(data_folder, checkpoint_folder)
