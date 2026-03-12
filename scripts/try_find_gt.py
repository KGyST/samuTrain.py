#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '.')

from src.db import get_random_case

print('=== Finding case with actual GT text ===')
for i in range(50):
    case = get_random_case(weighted=True)
    if case and case.get('gt_text') and case.get('gt_text').strip():
        print(f'Found GT case: ID={case["id"]}, gt_text="{case["gt_text"]}"')
        break
else:
    if i % 10 == 9:
        print(f'Checked {i+1} cases, no GT found yet...')

print('=== Summary ===')
# Count GT cases
import sqlite3
conn = sqlite3.connect('samu.db')
cursor = conn.execute('SELECT COUNT(*) FROM cases WHERE gt_text IS NOT NULL AND gt_text != ""')
count = cursor.fetchone()[0]
print(f'Total cases with non-empty GT: {count}')
conn.close()
