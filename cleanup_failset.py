import sys
sys.path.insert(0, 'src')
from db import Database

db = Database()
# Remove the temp_failset case
if db.delete_case_by_img_path('temp_failset\\test001.bin.png'):
  print("✅ Removed temp_failset case")

# Check remaining failset cases
cases = db.get_cases(limit=10, failset_only=True)
print('Remaining failset cases:')
for case in cases:
  print(f"  ID: {case['id']}, Path: {case['img_path']}, GT: '{case['gt_text']}', Failset: {case['is_failset']}")
