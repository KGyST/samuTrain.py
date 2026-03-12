#!/usr/bin/env python3
import os

# Check all subfolders for images
subfolders = ['1962', '64_case', 'new_case', 'single_case']

for folder in subfolders:
    folder_path = f"data/{folder}"
    if os.path.exists(folder_path):
        images = [f for f in os.listdir(folder_path) if f.endswith('.bin.png')]
        print(f"{folder}: {len(images)} images")
        if images:
            print(f"  Sample: {images[:3]}")
    else:
        print(f"{folder}: folder not found")
