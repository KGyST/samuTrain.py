#!/usr/bin/env python3
import os
import sqlite3

conn = sqlite3.connect('samu.db')
cursor = conn.execute('SELECT id, img_path FROM cases WHERE gt_text IS NOT NULL AND gt_text != "" LIMIT 5')
rows = cursor.fetchall()

print("=== Checking if GT case images exist ===")
for row in rows:
    img_path = f"data/{row[1]}"
    exists = os.path.exists(img_path)
    print(f'ID: {row[0]}, Path: {img_path}, Exists: {exists}')

conn.close()
