import sqlite3

conn = sqlite3.connect('samu.db')
cursor = conn.cursor()

cursor.execute("SELECT MIN(confidence), MAX(confidence) FROM cases")
min_conf, max_conf = cursor.fetchone()

cursor.execute("SELECT COUNT(*) FROM cases WHERE confidence != 0.0")
non_zero = cursor.fetchone()[0]

conn.close()

print(f"Min confidence: {min_conf}")
print(f"Max confidence: {max_conf}")
print(f"Records with confidence != 0.0: {non_zero}")
