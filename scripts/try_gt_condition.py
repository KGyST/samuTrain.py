#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '.')

from src.db import get_random_case

print("=== Testing GT text conditions ===")
for i in range(10):
    case = get_random_case(weighted=True)
    if case:
        gt_text = case.get('gt_text')
        has_gt = bool(gt_text)
        has_gt_and_not_empty = gt_text is not None and gt_text != ''
        
        print(f"Case ID={case['id']}: gt_text='{gt_text}', bool(gt_text)={has_gt}, gt_text and not empty={has_gt_and_not_empty}")
        
        if has_gt_and_not_empty:
            print(f"  -> This case SHOULD be evaluated!")
            break
