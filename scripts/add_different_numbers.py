#!/usr/bin/env python3

import os
import sys
import shutil
import random

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def add_different_numbers_to_dataset(dataset_name="1962", num_images=15):
  """Add images with completely different numbers to test dataset"""
  dataset_path = f"data/{dataset_name}"
  
  if not os.path.exists(dataset_path):
    print(f"❌ Dataset {dataset_name} not found")
    return False
  
  print(f"🎲 Adding {num_images} images with different numbers to {dataset_name} dataset...")
  
  # Find existing images to copy as templates
  existing_images = [f for f in os.listdir(dataset_path) if f.endswith('.bin.png')]
  if not existing_images:
    print(f"❌ No existing images found in {dataset_name}")
    return False
  
  # Different numbers to add as ground truth (completely different from 1962)
  different_numbers = [
    "1960", "1961", "1963", "1964", "1965", "1966", "1967", "1968", "1969",
    "1970", "1975", "1980", "1985", "1990", "1995", "2000", "2005", "2010",
    "1234", "5678", "9999", "1111", "2222", "3333", "4444", "5555", "6666",
    "7777", "8888", "0000", "2020", "2021", "2022", "2023", "2024", "2025"
  ]
  
  # Select random existing images as templates
  template_images = random.sample(existing_images, min(num_images, len(existing_images)))
  
  added_count = 0
  for i, template_img in enumerate(template_images):
    # Create new image name
    number = different_numbers[i % len(different_numbers)]
    new_name = f"different_{i+1:03d}_{os.path.basename(template_img)}"
    src_path = os.path.join(dataset_path, template_img)
    dst_path = os.path.join(dataset_path, new_name)
    
    # Copy the image
    shutil.copy2(src_path, dst_path)
    
    # Create corresponding GT file with different number
    gt_path = dst_path.replace('.bin.png', '.gt.txt')
    with open(gt_path, 'w', encoding='utf-8') as f:
      f.write(number)
    
    added_count += 1
    print(f"  ✅ Copied {template_img} -> {new_name}")
    print(f"  📝 Created GT: {gt_path} = '{number}'")
  
  print(f"🎉 Added {added_count} images with different numbers to {dataset_name} dataset")
  print(f"📝 These images have completely different GT text for testing")
  return True

if __name__ == "__main__":
  dataset_name = sys.argv[1] if len(sys.argv) > 1 else "1962"
  num_images = int(sys.argv[2]) if len(sys.argv) > 2 else 15
  
  success = add_different_numbers_to_dataset(dataset_name, num_images)
  if success:
    print(f"✅ Successfully added images with different numbers to {dataset_name}")
    print("💡 Restart the server to see the new images in the UI")
  else:
    print("❌ Failed to add images with different numbers")
