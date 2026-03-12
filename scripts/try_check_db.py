#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('samu.db')
cursor = conn.execute('SELECT id, img_path, ocr_text, confidence, gt_text FROM cases ORDER BY id DESC LIMIT 5')
rows = cursor.fetchall()

print("=== Recent Database Entries ===")
for row in rows:
    print(f'ID: {row[0]}, Path: {row[1]}, OCR: "{row[2]}", Conf: {row[3]}, GT: "{row[4]}"')

print("\n=== Cases with GT text ===")
cursor = conn.execute('SELECT COUNT(*) FROM cases WHERE gt_text IS NOT NULL AND gt_text != ""')
count = cursor.fetchone()[0]
print(f"Total cases with GT: {count}")

print("\n=== Sample GT cases ===")
cursor = conn.execute('SELECT id, img_path, gt_text FROM cases WHERE gt_text IS NOT NULL AND gt_text != "" LIMIT 3')
for row in cursor.fetchall():
    print(f'ID: {row[0]}, Path: {row[1]}, GT: "{row[2]}"')

conn.close()
