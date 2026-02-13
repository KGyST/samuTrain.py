import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class Database:
  def __init__(self, db_path: str = "samu.db"):
    self.db_path = db_path
    self.init_database()
  
  def init_database(self):
    with sqlite3.connect(self.db_path) as conn:
      conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          img_path TEXT NOT NULL,
          ocr_text TEXT NOT NULL,
          gt_text TEXT,
          confidence REAL NOT NULL,
          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
          is_corrected BOOLEAN DEFAULT FALSE,
          is_failset BOOLEAN DEFAULT FALSE
        )
      """)
      conn.commit()
  
  def insert_case(self, img_path: str, ocr_text: str, confidence: float, 
                  gt_text: Optional[str] = None, is_failset: bool = False) -> int:
    with sqlite3.connect(self.db_path) as conn:
      cursor = conn.execute("""
        INSERT INTO cases (img_path, ocr_text, gt_text, confidence, is_failset)
        VALUES (?, ?, ?, ?, ?)
      """, (img_path, ocr_text, gt_text, confidence, is_failset))
      conn.commit()
      return cursor.lastrowid
  
  def get_cases(self, limit: int = 100, failset_only: bool = False) -> List[Dict[str, Any]]:
    with sqlite3.connect(self.db_path) as conn:
      conn.row_factory = sqlite3.Row
      query = "SELECT * FROM cases"
      params = []
      
      if failset_only:
        query += " WHERE is_failset = TRUE"
      
      query += " ORDER BY timestamp DESC LIMIT ?"
      params.append(limit)
      
      cursor = conn.execute(query, params)
      return [dict(row) for row in cursor.fetchall()]
  
  def get_case_by_id(self, case_id: int) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(self.db_path) as conn:
      conn.row_factory = sqlite3.Row
      cursor = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,))
      row = cursor.fetchone()
      return dict(row) if row else None
  
  def update_case_correction(self, case_id: int, corrected_text: str) -> bool:
    with sqlite3.connect(self.db_path) as conn:
      case = self.get_case_by_id(case_id)
      if not case:
        return False
      
      img_path = case['img_path']
      
      # Update .gt.txt file on disk
      try:
        gt_path = img_path.rsplit('.', 1)[0] + '.gt.txt'
        with open(gt_path, 'w', encoding='utf-8') as f:
          f.write(corrected_text)
      except Exception as e:
        print(f"Failed to update GT file {gt_path}: {e}")
        return False
      
      # Update database
      conn.execute("""
        UPDATE cases 
        SET gt_text = ?, is_corrected = TRUE 
        WHERE id = ?
      """, (corrected_text, case_id))
      conn.commit()
      return True
  
  def get_statistics(self) -> Dict[str, Any]:
    with sqlite3.connect(self.db_path) as conn:
      conn.row_factory = sqlite3.Row
      
      # Total cases
      total_cursor = conn.execute("SELECT COUNT(*) as count FROM cases")
      total_cases = total_cursor.fetchone()['count']
      
      # Failset cases
      failset_cursor = conn.execute("SELECT COUNT(*) as count FROM cases WHERE is_failset = TRUE")
      failset_cases = failset_cursor.fetchone()['count']
      
      # Average confidence
      avg_cursor = conn.execute("SELECT AVG(confidence) as avg_conf FROM cases")
      avg_confidence = avg_cursor.fetchone()['avg_conf'] or 0.0
      
      # Recent activity (last hour)
      recent_cursor = conn.execute("""
        SELECT COUNT(*) as count FROM cases 
        WHERE timestamp > datetime('now', '-1 hour')
      """)
      recent_activity = recent_cursor.fetchone()['count']
      
      # Failset ratio
      failset_ratio = (failset_cases / total_cases * 100) if total_cases > 0 else 0.0
      
      return {
        'total_cases': total_cases,
        'failset_cases': failset_cases,
        'avg_confidence': avg_confidence,
        'recent_activity': recent_activity,
        'failset_ratio': failset_ratio
      }
