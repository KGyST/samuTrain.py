import sys
sys.path.insert(0, 'src')
from db import Database

db = Database()
cases = db.get_cases(limit=10, failset_only=True)
print('Failset cases in database:')
for case in cases:
  print(f"  ID: {case['id']}, Path: {case['img_path']}, GT: '{case['gt_text']}', Failset: {case['is_failset']}")
