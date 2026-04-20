import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


def _resolve_case_image_file(img_path: str) -> str:
  """Resolve on-disk path for a case image using SAMUTRAIN_DATA_FOLDER.

  Supports: path under data folder, path relative to data parent (e.g. 64_case/foo.png),
  and basename-only rows when the file lives in a sibling dataset folder (e.g. single_case
  env but images under data/64_case/).
  """
  sep_img = img_path.replace("/", os.sep)
  data_folder = os.path.normpath(
    os.path.abspath(os.environ.get("SAMUTRAIN_DATA_FOLDER", "data"))
  )
  parent_data = os.path.dirname(data_folder)

  direct = os.path.join(data_folder, sep_img)
  if os.path.isfile(direct):
    return direct

  if parent_data and os.path.isdir(parent_data):
    under_parent = os.path.normpath(os.path.join(parent_data, sep_img))
    if os.path.isfile(under_parent):
      return under_parent

  base_only = os.path.basename(sep_img)
  if parent_data and os.path.isdir(parent_data) and base_only == sep_img:
    try:
      for name in sorted(os.listdir(parent_data)):
        subdir = os.path.join(parent_data, name)
        if not os.path.isdir(subdir):
          continue
        candidate = os.path.join(subdir, base_only)
        if os.path.isfile(candidate):
          return candidate
    except OSError:
      pass

  return direct


class Database:
  def __init__(self, db_path: str = "samu.db"):
    self.db_path = db_path
    self.init_database()
  
  def init_database(self):
    print(f"🔧 Initializing database at: {self.db_path}")
    try:
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
            is_failset BOOLEAN DEFAULT FALSE,
            updated_flag BOOLEAN DEFAULT FALSE,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
          )
        """)
        
        # Check if training_sessions exists and has correct structure
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_sessions'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
          # Check if table has the expected columns
          cursor = conn.execute("PRAGMA table_info(training_sessions)")
          columns = [row[1] for row in cursor.fetchall()]
          
          expected_columns = ['session_id', 'data_folder', 'checkpoint_folder', 'network', 'backup_folder', 'model_folder', 'status', 'chars_count', 'started_at', 'completed_at', 'error_message']
          
          if not all(col in columns for col in expected_columns):
            print("⚠️ Existing training_sessions table has different structure, creating new table")
            conn.execute("DROP TABLE training_sessions")
            table_exists = False
        
        if not table_exists:
          conn.execute("""
            CREATE TABLE training_sessions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT UNIQUE NOT NULL,
              data_folder TEXT NOT NULL,
              checkpoint_folder TEXT NOT NULL,
              network TEXT,
              backup_folder TEXT,
              model_folder TEXT,
              status TEXT DEFAULT 'started',
              chars_count INTEGER DEFAULT 0,
              started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              completed_at DATETIME,
              error_message TEXT
            )
          """)
        
        conn.commit()
        print("✅ Database tables created successfully")
        
        # Check if cases table has new columns and migrate if needed
        cursor = conn.execute("PRAGMA table_info(cases)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'updated_flag' not in columns:
          print("🔄 Migrating cases table to add update flags...")
          conn.execute("ALTER TABLE cases ADD COLUMN updated_flag BOOLEAN DEFAULT FALSE")
          conn.execute("ALTER TABLE cases ADD COLUMN last_updated DATETIME DEFAULT CURRENT_TIMESTAMP")
          conn.commit()
          print("✅ Cases table migration completed")
    except Exception as e:
      print(f"❌ Database initialization error: {e}")
  
  def batch_insert_cases(self, cases_data: List[Dict[str, Any]]) -> int:
    """Insert multiple cases in a single transaction for better performance"""
    if not cases_data:
      return 0
    
    try:
      with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        
        # Prepare the insert statement
        insert_stmt = """
          INSERT INTO cases (img_path, ocr_text, model_prediction, gt_text, confidence, is_corrected, is_failset)
          VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        # Prepare batch data
        batch_data = []
        for case in cases_data:
          batch_data.append((
            case['img_path'],
            case['ocr_text'],
            case.get('model_prediction', case['ocr_text']),
            case.get('gt_text'),
            case['confidence'],
            case.get('is_corrected', False),
            case.get('is_failset', False)
          ))
        
        # Execute batch insert
        cursor.executemany(insert_stmt, batch_data)
        conn.commit()
        
        inserted_count = cursor.rowcount
        print(f"✅ Batch inserted {inserted_count} new cases")
        return inserted_count
        
    except Exception as e:
      print(f"❌ Batch insert error: {e}")
      return 0
  
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
  
  def get_random_case(self, weighted: bool = True) -> Optional[Dict[str, Any]]:
    """Get a random case, optionally weighted towards failset cases"""
    with sqlite3.connect(self.db_path) as conn:
      conn.row_factory = sqlite3.Row
      
      if weighted:
        # Weighted selection: 70% trainset, 30% failset
        cursor = conn.execute("SELECT COUNT(*) as count FROM cases WHERE is_failset = FALSE")
        train_count = cursor.fetchone()['count']
        
        cursor = conn.execute("SELECT COUNT(*) as count FROM cases WHERE is_failset = TRUE")
        fail_count = cursor.fetchone()['count']
        
        total = train_count + fail_count
        if total == 0:
          return None
          
        import random
        use_failset = (random.random() < 0.3) and (fail_count > 0) or (train_count == 0)
        
        if use_failset:
          cursor = conn.execute("SELECT * FROM cases WHERE is_failset = TRUE ORDER BY RANDOM() LIMIT 1")
        else:
          cursor = conn.execute("SELECT * FROM cases WHERE is_failset = FALSE ORDER BY RANDOM() LIMIT 1")
      else:
        cursor = conn.execute("SELECT * FROM cases ORDER BY RANDOM() LIMIT 1")
      
      row = cursor.fetchone()
      return dict(row) if row else None
  
  def get_all_failset_cases(self) -> List[Dict[str, Any]]:
    """Get all cases from failset that have GT text for training"""
    with sqlite3.connect(self.db_path) as conn:
      conn.row_factory = sqlite3.Row
      cursor = conn.execute("""
        SELECT * FROM cases 
        WHERE is_failset = TRUE AND gt_text IS NOT NULL AND gt_text != ''
        ORDER BY timestamp DESC
      """)
      return [dict(row) for row in cursor.fetchall()]

  def get_case(self, case_id: int) -> Optional[Dict[str, Any]]:
    """Alias for get_case_by_id for backward compatibility"""
    return self.get_case_by_id(case_id)

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
        data_folder = os.environ.get('SAMUTRAIN_DATA_FOLDER', 'data')
        if img_path.endswith('.bin.png'):
            gt_filename = img_path.replace('.bin.png', '.gt.txt')
        else:
            gt_filename = img_path.rsplit('.', 1)[0] + '.gt.txt'
        base_dir = data_folder if '/' not in img_path else os.path.dirname(data_folder)
        gt_path = os.path.join(base_dir, gt_filename)
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
    """Update the failset status of a case with detailed logging"""
    try:
      with sqlite3.connect(self.db_path) as conn:
        # Get current status first
        cursor = conn.execute("SELECT is_failset FROM cases WHERE id = ?", (case_id,))
        result = cursor.fetchone()
        if not result:
          print(f"❌ Case {case_id} not found")
          return False
        
        was_failset = bool(result[0])
        
        # Only update if status actually changed
        if was_failset != is_failset:
          conn.execute("UPDATE cases SET is_failset = ? WHERE id = ?", (is_failset, case_id))
          conn.commit()
          
          if was_failset and not is_failset:
            print(f"✅ Case {case_id}: was in FAILSET, ok, moved to TRAINSET")
          elif not was_failset and is_failset:
            print(f"❌ Case {case_id}: was in TRAINSET, failed, moved to FAILSET")
        else:
          # Status didn't change
          if was_failset:
            print(f"❌ Case {case_id}: was in FAILSET, fails, stays in FAILSET")
          else:
            print(f"✅ Case {case_id}: was in TRAINSET, ok, stays in TRAINSET")
        
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

  def get_updated_cases(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Get cases with update flags set for frontend notifications"""
    try:
      with sqlite3.connect(self.db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        if since:
          cursor = conn.execute("""
            SELECT * FROM cases 
            WHERE updated_flag = TRUE AND last_updated > ?
            ORDER BY last_updated DESC
          """, (since,))
        else:
          cursor = conn.execute("""
            SELECT * FROM cases 
            WHERE updated_flag = TRUE
            ORDER BY last_updated DESC
          """)
        
        return [dict(row) for row in cursor.fetchall()]
        
    except Exception as e:
      print(f"❌ Get updated cases error: {e}")
      return []
  
  def clear_update_flags(self, case_ids: List[int]) -> bool:
    """Clear update flags after frontend reads them"""
    if not case_ids:
      return True
    
    try:
      with sqlite3.connect(self.db_path) as conn:
        placeholders = ','.join('?' * len(case_ids))
        conn.execute(f"""
          UPDATE cases 
          SET updated_flag = FALSE 
          WHERE id IN ({placeholders})
        """, case_ids)
        conn.commit()
        
        print(f"✅ Cleared update flags for {len(case_ids)} cases")
        return True
        
    except Exception as e:
      print(f"❌ Clear update flags error: {e}")
      return False
  
  def mark_cases_updated(self, case_ids: List[int]) -> bool:
    """Mark cases as updated for frontend notification"""
    if not case_ids:
      return True
    
    try:
      with sqlite3.connect(self.db_path) as conn:
        placeholders = ','.join('?' * len(case_ids))
        conn.execute(f"""
          UPDATE cases 
          SET updated_flag = TRUE, last_updated = CURRENT_TIMESTAMP
          WHERE id IN ({placeholders})
        """, case_ids)
        conn.commit()
        
        print(f"✅ Marked {len(case_ids)} cases as updated")
        return True
        
    except Exception as e:
      print(f"❌ Mark cases updated error: {e}")
      return False

  def get_training_cases(self, include_gt_only: bool = True) -> List[Dict[str, Any]]:
    """Get all cases for training with image paths (database-centric approach)"""
    try:
      with sqlite3.connect(self.db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        if include_gt_only:
          cursor = conn.execute("""
            SELECT id, img_path, gt_text, ocr_text, confidence
            FROM cases 
            WHERE gt_text IS NOT NULL AND gt_text != ''
            ORDER BY id
          """)
        else:
          cursor = conn.execute("""
            SELECT id, img_path, gt_text, ocr_text, confidence
            FROM cases 
            ORDER BY id
          """)
        
        cases = []
        for row in cursor.fetchall():
          case = dict(row)
          case["full_img_path"] = _resolve_case_image_file(case["img_path"])
          cases.append(case)
        
        return cases
        
    except Exception as e:
      print(f"❌ Get training cases error: {e}")
      return []
  
  def update_predictions_batch(self, predictions: List[Dict[str, Any]]) -> bool:
    """Batch update OCR predictions and confidence from Calamari"""
    if not predictions:
      return True
    
    try:
      with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        
        # Prepare update statement
        update_stmt = """
          UPDATE cases 
          SET ocr_text = ?, model_prediction = ?, confidence = ?, updated_flag = TRUE, last_updated = CURRENT_TIMESTAMP
          WHERE id = ?
        """
        
        # Prepare batch data
        batch_data = []
        case_ids = []
        for pred in predictions:
          batch_data.append((
            pred['ocr_text'],
            pred.get('model_prediction', pred['ocr_text']),
            pred['confidence'],
            pred['case_id']
          ))
          case_ids.append(pred['case_id'])
        
        # Execute batch update
        cursor.executemany(update_stmt, batch_data)
        conn.commit()
        
        updated_count = cursor.rowcount
        print(f"✅ Batch updated {updated_count} predictions")
        return True
        
    except Exception as e:
      print(f"❌ Batch update predictions error: {e}")
      return False

  def get_statistics(self) -> Dict[str, Any]:
    """Get database statistics"""
    with sqlite3.connect(self.db_path) as conn:
      cursor = conn.execute("SELECT COUNT(*) FROM cases")
      total = cursor.fetchone()[0]
      
      cursor = conn.execute("SELECT COUNT(*) FROM cases WHERE is_failset = 1")
      failset = cursor.fetchone()[0]
      
      cursor = conn.execute("SELECT AVG(confidence) FROM cases")
      avg_row = cursor.fetchone()
      avg_conf = avg_row[0] if avg_row[0] else 0
      
      cursor = conn.execute(
        "SELECT COUNT(*) FROM cases WHERE timestamp >= datetime('now', '-1 hour')"
      )
      recent_activity = cursor.fetchone()[0]
      
      cursor = conn.execute("SELECT COUNT(*) FROM training_sessions")
      sessions = cursor.fetchone()[0]
      
      failset_ratio = (100.0 * failset / total) if total else 0.0
      return {
        "total_cases": total,
        "failset_cases": failset,
        "avg_confidence": round(avg_conf, 3),
        "recent_activity": recent_activity,
        "failset_ratio": failset_ratio,
        "training_sessions": sessions
      }
  
  def insert_training_session(self, session_id: str, data_folder: str, checkpoint_folder: str,
                            network: Optional[str] = None, backup_folder: Optional[str] = None,
                            chars_count: int = 0) -> int:
    """Insert a new training session"""
    with sqlite3.connect(self.db_path) as conn:
      cursor = conn.execute("""
        INSERT INTO training_sessions (session_id, data_folder, checkpoint_folder, network, backup_folder, chars_count)
        VALUES (?, ?, ?, ?, ?, ?)
      """, (session_id, data_folder, checkpoint_folder, network, backup_folder, chars_count))
      conn.commit()
      return cursor.lastrowid
  
  def update_training_session(self, session_id: str, status: str, model_folder: Optional[str] = None,
                            error_message: Optional[str] = None):
    """Update training session status"""
    with sqlite3.connect(self.db_path) as conn:
      if status == 'completed':
        conn.execute("""
          UPDATE training_sessions 
          SET status = ?, model_folder = ?, completed_at = CURRENT_TIMESTAMP 
          WHERE session_id = ?
        """, (status, model_folder, session_id))
      elif status == 'failed':
        conn.execute("""
          UPDATE training_sessions 
          SET status = ?, error_message = ?, completed_at = CURRENT_TIMESTAMP 
          WHERE session_id = ?
        """, (status, error_message, session_id))
      else:
        conn.execute("""
          UPDATE training_sessions 
          SET status = ? 
          WHERE session_id = ?
        """, (status, session_id))
      conn.commit()
  
  def get_training_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent training sessions"""
    with sqlite3.connect(self.db_path) as conn:
      cursor = conn.execute("""
        SELECT * FROM training_sessions 
        ORDER BY started_at DESC 
        LIMIT ?
      """, (limit,))
      
      columns = [desc[0] for desc in cursor.description]
      return [dict(zip(columns, row)) for row in cursor.fetchall()]

  def reset_database(self):
    """Reset database by dropping and recreating tables instead of deleting file"""
    print(f"🧹 Resetting database at: {self.db_path}")
    try:
      with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        # Drop existing tables
        cursor.execute("DROP TABLE IF EXISTS cases")
        print("🗑️  Dropped existing tables")
        
        # Recreate tables
        self.init_database()
        
        conn.commit()
        print("✅ Database reset complete (schema wiped, file kept)")
        return True
    except Exception as e:
      print(f"⚠️ Reset failed: {e}")
      return False

# Global database instance
db = Database()

# Global functions for backward compatibility
def get_db_connection():
    """Get database connection (for backward compatibility)"""
    return sqlite3.connect(db.db_path)

def init_database():
    """Initialize database (for backward compatibility)"""
    db.init_database()

def get_cases(limit: int = 100, failset_only: bool = False):
    return db.get_cases(limit, failset_only)

def get_case_by_id(case_id: int):
    return db.get_case_by_id(case_id)

def get_case(case_id: int):
    return db.get_case(case_id)

def get_case_by_img_path(img_path: str):
    return db.get_case_by_img_path(img_path)

def get_random_case(weighted: bool = True):
    return db.get_random_case(weighted)

def get_all_failset_cases():
    return db.get_all_failset_cases()

def insert_case(img_path: str, ocr_text: str, confidence: float, 
               gt_text: Optional[str] = None, is_failset: bool = False, 
               model_prediction: Optional[str] = None):
    return db.insert_case(img_path, ocr_text, confidence, gt_text, is_failset, model_prediction)

def update_case_correction(case_id: int, corrected_text: str):
    return db.update_case_correction(case_id, corrected_text)

def update_case_ocr_result(case_id: int, ocr_text: str, confidence: float):
    return db.update_case_ocr_result(case_id, ocr_text, confidence)

def update_case_gt_text(case_id: int, gt_text: str):
    return db.update_case_gt_text(case_id, gt_text)

def update_case_failset_status(case_id: int, is_failset: bool):
    return db.update_case_failset_status(case_id, is_failset)

def update_model_prediction(case_id: int, model_prediction: str):
    return db.update_model_prediction(case_id, model_prediction)

def delete_case_by_img_path(img_path: str):
    return db.delete_case_by_img_path(img_path)

def get_statistics():
  return db.get_statistics()

def insert_training_session(session_id: str, data_folder: str, checkpoint_folder: str,
                          network: Optional[str] = None, backup_folder: Optional[str] = None,
                          chars_count: int = 0) -> int:
  return db.insert_training_session(session_id, data_folder, checkpoint_folder, network, backup_folder, chars_count)

def update_training_session(session_id: str, status: str, model_folder: Optional[str] = None,
                          error_message: Optional[str] = None):
  return db.update_training_session(session_id, status, model_folder, error_message)

def get_training_sessions(limit: int = 10) -> List[Dict[str, Any]]:
  return db.get_training_sessions(limit)
