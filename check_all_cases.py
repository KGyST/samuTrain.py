import sys
sys.path.insert(0, 'src')
from db import Database

db = Database()
cases = db.get_cases(limit=10)
print('All cases in database:')
for case in cases:
  print(f"  ID: {case['id']}, Path: {case['img_path']}, GT: '{case['gt_text']}', OCR: '{case['ocr_text']}', Failset: {case['is_failset']}")
