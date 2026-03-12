#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '.')

from src.db import get_random_case

print("=== Testing get_random_case function ===")
for i in range(5):
    case = get_random_case(weighted=True)
    if case:
        print(f"Case {i+1}: ID={case['id']}, img_path={case['img_path']}, gt_text='{case.get('gt_text', 'None')}', has_gt={bool(case.get('gt_text'))}")
    else:
        print(f"Case {i+1}: None returned")
