#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '.')

from src.db import Database, get_random_case
from src.bridge import OCRBridge

print("=== Testing evaluation loop logic ===")

# Use same logic as evaluation loop
db = Database()
ocr_bridge = OCRBridge()

for i in range(5):
    print(f"\nAttempt {i+1}:")
    case = get_random_case(weighted=True)
    print(f"  get_random_case returned: {case is not None}")
    
    if case:
        gt_text = case.get('gt_text')
        print(f"  GT text: '{gt_text}'")
        print(f"  bool(gt_text): {bool(gt_text)}")
        print(f"  gt_text.strip(): '{gt_text.strip() if gt_text else None}'")
        print(f"  bool(gt_text.strip()): {bool(gt_text.strip()) if gt_text else False}")
        
        if gt_text and gt_text.strip():
            print(f"  -> Should evaluate this case!")
            break
        else:
            print(f"  -> Skipping (no GT)")
    else:
        print(f"  -> No case returned")
