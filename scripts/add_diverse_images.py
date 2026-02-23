#!/usr/bin/env python3

import os
import sys
import shutil
import random

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def add_diverse_images_to_dataset(dataset_name="1962", num_images=10):
  """Add images with different 19xx content to test dataset"""
  dataset_path = f"data/{dataset_name}"
  
  if not os.path.exists(dataset_path):
    print(f"❌ Dataset {dataset_name} not found")
    return False
  
  print(f"🎲 Adding {num_images} diverse 19xx images to {dataset_name} dataset...")
  
  # Find existing images to copy as templates
  existing_images = [f for f in os.listdir(dataset_path) if f.endswith('.bin.png')]
  if not existing_images:
    print(f"❌ No existing images found in {dataset_name}")
    return False
  
  # Different 19xx years to add as ground truth
  diverse_years = ["1955", "1958", "1959", "1960", "1961", "1963", "1964", "1965", "1966", "1967"]
  
  # Select random existing images as templates
  template_images = random.sample(existing_images, min(num_images, len(existing_images)))
  
  added_count = 0
  for i, template_img in enumerate(template_images):
    # Create new image name
    year = diverse_years[i % len(diverse_years)]
    new_name = f"diverse_{i+1:03d}_{os.path.basename(template_img)}"
    src_path = os.path.join(dataset_path, template_img)
    dst_path = os.path.join(dataset_path, new_name)
    
    # Copy the image
    shutil.copy2(src_path, dst_path)
    
    # Create corresponding GT file with different year
    gt_path = dst_path.replace('.bin.png', '.gt.txt')
    with open(gt_path, 'w', encoding='utf-8') as f:
      f.write(year)
    
    added_count += 1
    print(f"  ✅ Copied {template_img} -> {new_name}")
    print(f"  📝 Created GT: {gt_path} = '{year}'")
  
  print(f"🎉 Added {added_count} diverse images to {dataset_name} dataset")
  print(f"📝 These images have different GT text (19xx years) for testing")
  return True

if __name__ == "__main__":
  dataset_name = sys.argv[1] if len(sys.argv) > 1 else "1962"
  num_images = int(sys.argv[2]) if len(sys.argv) > 2 else 10
  
  success = add_diverse_images_to_dataset(dataset_name, num_images)
  if success:
    print(f"✅ Successfully added diverse images to {dataset_name}")
    print("💡 Restart the server to see the new images in the UI")
  else:
    print("❌ Failed to add diverse images")
