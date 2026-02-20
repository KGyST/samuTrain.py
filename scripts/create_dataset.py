#!/usr/bin/env python3
import os
import shutil
from collections import defaultdict
import re

def scan_archive_for_patterns():
  """Scan archive/trainsets for common ground truth patterns"""
  archive_path = os.path.join(os.path.dirname(__file__), '..', 'archive', 'trainsets')
  gt_counts = defaultdict(list)
  year_pattern = re.compile(r'\b(19|20)\d{2}\b')
  
  print(f"Scanning {archive_path}...")
  
  if not os.path.exists(archive_path):
    print(f"Archive path not found: {archive_path}")
    return gt_counts
  
  # Scan all trainset directories
  for trainset_dir in os.listdir(archive_path):
    trainset_path = os.path.join(archive_path, trainset_dir)
    if not os.path.isdir(trainset_path):
      continue
    
    # Skip hidden directories
    if trainset_dir.startswith('.'):
      continue
    
    print(f"Processing {trainset_dir}...")
    
    # Find all .gt.txt files
    for filename in os.listdir(trainset_path):
      if filename.endswith('.gt.txt'):
        gt_path = os.path.join(trainset_path, filename)
        try:
          with open(gt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
              gt_counts[content].append((gt_path, filename))
        except Exception as e:
          print(f"Error reading {gt_path}: {e}")
  
  return gt_counts

def find_best_target_set(gt_counts, target_size=100):
  """Find the best target set for creating a dataset"""
  # First, look for exact matches with high counts
  exact_matches = [(text, len(files)) for text, files in gt_counts.items() if len(files) >= target_size]
  if exact_matches:
    exact_matches.sort(key=lambda x: x[1], reverse=True)
    return exact_matches[0][0], gt_counts[exact_matches[0][0]]
  
  # If no exact matches, look for year patterns
  year_groups = defaultdict(list)
  for text, files in gt_counts.items():
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    if year_match:
      year = year_match.group()
      year_groups[year].append((text, files))
  
  # Find years with most examples
  year_totals = [(year, sum(len(files) for _, files in groups)) for year, groups in year_groups.items()]
  year_totals.sort(key=lambda x: x[1], reverse=True)
  
  if year_totals and year_totals[0][1] >= target_size:
    best_year = year_totals[0][0]
    all_files = []
    for text, files in year_groups[best_year]:
      all_files.extend(files)
    return f"year_{best_year}", all_files[:target_size]
  
  # Fallback to the largest available set
  if gt_counts:
    best_text = max(gt_counts.keys(), key=lambda k: len(gt_counts[k]))
    return best_text, gt_counts[best_text]
  
  return None, []

def create_dataset(target_name, source_files):
  """Create a new dataset under data/"""
  data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
  dataset_path = os.path.join(data_path, target_name)
  
  if os.path.exists(dataset_path):
    print(f"Dataset {target_name} already exists")
    return
  
  os.makedirs(dataset_path, exist_ok=True)
  
  copied_count = 0
  missing_images = 0
  print(f"Attempting to copy {len(source_files)} files...")
  
  for i, (gt_path, gt_filename) in enumerate(source_files):
    # Create unique filename by prefixing with trainset directory
    trainset_dir = os.path.basename(os.path.dirname(gt_path))
    unique_gt_filename = f"{trainset_dir}_{gt_filename}"
    new_gt_path = os.path.join(dataset_path, unique_gt_filename)
    
    try:
      shutil.copy2(gt_path, new_gt_path)
    except Exception as e:
      print(f"Error copying GT file {gt_path}: {e}")
      continue
    
    # Copy the corresponding image file
    img_filename = gt_filename.replace('.gt.txt', '.bin.png')
    unique_img_filename = f"{trainset_dir}_{img_filename}"
    img_path = os.path.join(os.path.dirname(gt_path), img_filename)
    if os.path.exists(img_path):
      new_img_path = os.path.join(dataset_path, unique_img_filename)
      try:
        shutil.copy2(img_path, new_img_path)
        copied_count += 1
      except Exception as e:
        print(f"Error copying image file {img_path}: {e}")
    else:
      missing_images += 1
      if missing_images <= 5:  # Only show first 5 missing images
        print(f"Image file not found: {img_path}")
  
  print(f"Created dataset {target_name} with {copied_count} examples")
  print(f"Missing images: {missing_images}")

def main():
  print("Starting dataset creation...")
  
  # Scan for patterns
  gt_counts = scan_archive_for_patterns()
  
  if not gt_counts:
    print("No ground truth files found")
    return
  
  print(f"Found {len(gt_counts)} unique ground truth texts")
  
  # Show some statistics
  sorted_counts = sorted(gt_counts.items(), key=lambda x: len(x[1]), reverse=True)
  print("\nTop 10 most common texts:")
  for i, (text, files) in enumerate(sorted_counts[:10]):
    print(f"{i+1}. '{text}' ({len(files)} examples)")
  
  # Find best target set
  target_name, source_files = find_best_target_set(gt_counts)
  
  if not source_files:
    print("No suitable dataset found")
    return
  
  print(f"\nSelected target: {target_name}")
  print(f"Found {len(source_files)} examples")
  
  # Create the dataset
  create_dataset(target_name, source_files)
  
  print("Dataset creation complete!")

if __name__ == "__main__":
  main()
