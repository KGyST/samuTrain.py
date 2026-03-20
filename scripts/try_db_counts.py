import sqlite3

conn = sqlite3.connect('samu.db')
cursor = conn.cursor()

total = cursor.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
corrected = cursor.execute("SELECT COUNT(*) FROM cases WHERE is_corrected = 1").fetchone()[0]
not_corrected = cursor.execute("SELECT COUNT(*) FROM cases WHERE is_corrected = 0").fetchone()[0]

conn.close()

print(f"Total records: {total}")
print(f"Corrected: {corrected}")
print(f"Not corrected: {not_corrected}")
