import sys
sys.path.insert(0, 'src')
from db import Database
import sqlite3

# Clear all cases directly
with sqlite3.connect("samu.db") as conn:
  conn.execute("DELETE FROM cases")
  conn.commit()
  print("✅ Cleared all cases from database")

# Verify
db = Database()
cases = db.get_cases(limit=10)
print(f"Cases after clearing: {len(cases)}")
