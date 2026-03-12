#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('samu.db')
cursor = conn.execute('SELECT id, img_path, ocr_text, confidence, gt_text FROM cases WHERE id = 67')
row = cursor.fetchone()
if row:
    print(f'ID: {row[0]}, Path: {row[1]}, OCR: "{row[2]}", Conf: {row[3]}, GT: "{row[4]}"')
else:
    print('Case 67 not found')
conn.close()
