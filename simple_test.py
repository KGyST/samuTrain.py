#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from db import db

print("=== Database Test ===")
cases = db.get_cases()
print(f"Found {len(cases)} cases")

for case in cases:
    print(f"ID: {case['id']}, Path: {case['img_path']}, OCR: {case['ocr_text']}, GT: {case['gt_text']}")

print("\n=== Statistics Test ===")
stats = db.get_statistics()
print(f"Stats: {stats}")
