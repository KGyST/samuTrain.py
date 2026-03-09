# What is this file for: Script to collect all unique characters from ground truth texts in a data folder and print them.
# When it is created: 2026-03-04

import os
import sys
import glob

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

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python try_collect_chars.py <data_folder>")
    sys.exit(1)
  data_folder = sys.argv[1]
  chars = collect_chars(data_folder)
  print("Unique characters:", ''.join(chars))
