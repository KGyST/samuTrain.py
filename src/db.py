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
          model_prediction TEXT,
          gt_text TEXT,
          confidence REAL NOT NULL,
          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
          is_corrected BOOLEAN DEFAULT FALSE,
          is_failset BOOLEAN DEFAULT FALSE
        )
      """)
      conn.commit()
  
  def insert_case(self, img_path: str, ocr_text: str, confidence: float, 
                  gt_text: Optional[str] = None, is_failset: bool = False, 
                  model_prediction: Optional[str] = None) -> int:
    with sqlite3.connect(self.db_path) as conn:
      cursor = conn.execute("""
        INSERT INTO cases (img_path, ocr_text, model_prediction, gt_text, confidence, is_failset)
        VALUES (?, ?, ?, ?, ?, ?)
      """, (img_path, ocr_text, model_prediction, gt_text, confidence, is_failset))
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
  
  def get_case_by_img_path(self, img_path: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(self.db_path) as conn:
      conn.row_factory = sqlite3.Row
      cursor = conn.execute("SELECT * FROM cases WHERE img_path = ?", (img_path,))
      row = cursor.fetchone()
      return dict(row) if row else None

  def delete_case_by_img_path(self, img_path: str) -> bool:
    """Delete case by image path"""
    with sqlite3.connect(self.db_path) as conn:
      cursor = conn.execute("DELETE FROM cases WHERE img_path = ?", (img_path,))
      conn.commit()
      return cursor.rowcount > 0

  def update_case_correction(self, case_id: int, corrected_text: str) -> bool:
    print(f"=== CORRECTION REQUEST ===")
    print(f"Case ID: {case_id}")
    print(f"Corrected text: '{corrected_text}'")
    
    with sqlite3.connect(self.db_path) as conn:
      case = self.get_case_by_id(case_id)
      if not case:
        print(f"ERROR: Case {case_id} not found")
        return False
      
      img_path = case['img_path']
      print(f"Image path from database: {img_path}")
      
      # Handle static URLs - convert to actual file path
      if img_path.startswith('/static/'):
        # Get data folder from environment or default
        data_folder = os.environ.get('SAMUTRAIN_DATA_FOLDER', 'data')
        filename = img_path.replace('/static/', '')
        # Convert test001.bin.png to test001.gt.txt
        # Remove .bin.png and change to .gt.txt
        print(f"Original filename: {filename}")
        gt_filename = filename.replace('.bin.png', '.gt.txt')
        gt_path = os.path.join(data_folder, gt_filename)
        print(f"Converted static URL to GT path: {gt_path}")
        print(f"GT filename: {gt_filename}")
      else:
        # Original file path - convert .bin.png to .gt.txt
        if img_path.endswith('.bin.png'):
            gt_path = img_path.replace('.bin.png', '.gt.txt')
        else:
            gt_path = img_path.rsplit('.', 1)[0] + '.gt.txt'
        print(f"Using original GT path: {gt_path}")
      
      # Update .gt.txt file on disk
      try:
        print(f"Attempting to write to GT file: {gt_path}")
        print(f"File exists before write: {os.path.exists(gt_path)}")
        print(f"Directory exists: {os.path.exists(os.path.dirname(gt_path))}")
        
        with open(gt_path, 'w', encoding='utf-8') as f:
          f.write(corrected_text)
        
        print(f"Successfully wrote '{corrected_text}' to {gt_path}")
        print(f"File exists after write: {os.path.exists(gt_path)}")
        
        # Verify content was written
        with open(gt_path, 'r', encoding='utf-8') as f:
          verify_content = f.read()
        print(f"Verification - file content: '{verify_content}'")
        
      except Exception as e:
        print(f"Failed to update GT file {gt_path}: {e}")
        print(f"Exception type: {type(e).__name__}")
        return False
      
      # Update database - store user correction as both GT and OCR
      conn.execute("""
        UPDATE cases 
        SET gt_text = ?, ocr_text = ?, is_corrected = TRUE 
        WHERE id = ?
      """, (corrected_text, corrected_text, case_id))
      conn.commit()
      print(f"Database updated for case {case_id}")
      print(f"=== CORRECTION COMPLETED ===")
      return True
  
  def update_case_ocr_result(self, case_id: int, ocr_text: str, confidence: float) -> bool:
    """Update OCR result and confidence for a case"""
    try:
      with sqlite3.connect(self.db_path) as conn:
        conn.execute("UPDATE cases SET ocr_text = ?, confidence = ? WHERE id = ?", 
                    (ocr_text, float(confidence), case_id))
        conn.commit()
        return True
    except Exception as e:
      print(f"Error updating OCR result: {e}")
      return False

  def update_case_gt_text(self, case_id: int, gt_text: str) -> bool:
    """Update GT text in database only (does not write to GT file)"""
    try:
      with sqlite3.connect(self.db_path) as conn:
        conn.execute("UPDATE cases SET gt_text = ? WHERE id = ?", (gt_text, case_id))
        conn.commit()
        return True
    except Exception as e:
      print(f"Error updating GT text: {e}")
      return False

  def update_case_failset_status(self, case_id: int, is_failset: bool) -> bool:
    """Update the failset status of a case"""
    try:
      with sqlite3.connect(self.db_path) as conn:
        conn.execute("UPDATE cases SET is_failset = ? WHERE id = ?", (is_failset, case_id))
        conn.commit()
        return True
    except Exception as e:
      print(f"Error updating failset status: {e}")
      return False

  def update_model_prediction(self, case_id: int, model_prediction: str) -> bool:
    """Update the model_prediction field only"""
    try:
      with sqlite3.connect(self.db_path) as conn:
        conn.execute("UPDATE cases SET model_prediction = ? WHERE id = ?", (model_prediction, case_id))
        conn.commit()
        return True
    except Exception as e:
      print(f"Error updating model prediction: {e}")
      return False

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
