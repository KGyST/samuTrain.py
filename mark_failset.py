import sys
sys.path.insert(0, 'src')
from db import Database
import sqlite3

# Mark the case as failset so auto-training will trigger
with sqlite3.connect("samu.db") as conn:
  cursor = conn.cursor()
  cursor.execute("UPDATE cases SET is_failset = 1 WHERE img_path = ?", ('test001.bin.png',))
  conn.commit()
  print("✅ Marked case as failset")

# Verify
db = Database()
cases = db.get_cases(limit=10, failset_only=True)
print('Failset cases:')
for case in cases:
  print(f"  ID: {case['id']}, Path: {case['img_path']}, GT: '{case['gt_text']}', OCR: '{case['ocr_text']}', Failset: {case['is_failset']}")
