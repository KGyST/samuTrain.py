import sys
sys.path.insert(0, 'src')
from db import Database
import sqlite3

# Update the GT text directly
with sqlite3.connect("samu.db") as conn:
  cursor = conn.cursor()
  cursor.execute("UPDATE cases SET gt_text = ? WHERE img_path = ?", ('244.', 'test001.bin.png'))
  conn.commit()
  print("✅ Fixed GT text in database")

# Verify
db = Database()
cases = db.get_cases(limit=10)
print('Cases after fix:')
for case in cases:
  print(f"  ID: {case['id']}, Path: {case['img_path']}, GT: '{case['gt_text']}', OCR: '{case['ocr_text']}', Failset: {case['is_failset']}")
