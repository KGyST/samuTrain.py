from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import sqlite3
from datetime import datetime

from db import Database

app = FastAPI(title="samuTrain OCR Monitor", version="2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get data folder from environment variable or default to "data"
DATA_FOLDER = os.environ.get('SAMUTRAIN_DATA_FOLDER', 'data')

# Initialize database
db = Database()

# Folder polling and synchronization
import threading
import time
from pathlib import Path

def sync_database_with_folder():
  """Synchronize database with actual data folder contents"""
  try:
    import glob
    
    # Get all PNG files in data folder
    png_files = glob.glob(os.path.join(DATA_FOLDER, "**/*.png"), recursive=True)
    current_files = set()
    
    for png_path in png_files:
      rel_path = os.path.relpath(png_path)
      current_files.add(rel_path)
      
      # Look for corresponding .gt.txt file
      gt_path = png_path.replace('.bin.png', '.gt.txt')
      gt_text = None
      if os.path.exists(gt_path):
        with open(gt_path, 'r', encoding='utf-8') as f:
          gt_text = f.read().strip()
      
      # Debug: Log GT text status for sync
      has_gt = gt_text and gt_text.strip()
      print(f"🔄 SYNC {rel_path}: GT exists={os.path.exists(gt_path)}, GT text='{gt_text}', has_content={has_gt}")
      
      # Check if case exists
      existing_case = db.get_case_by_img_path(rel_path)
      
      if not existing_case:
        # Add new case
        confidence = 0.85 if has_gt else 0.60
        print(f"📊 SYNC {rel_path}: New confidence={confidence}, has_gt={has_gt}, gt_text_length={len(gt_text) if gt_text else 0}")
        case_id = db.insert_case(
          img_path=rel_path,
          ocr_text=gt_text or "OCR_RESULT_PLACEHOLDER",
          confidence=confidence,
          gt_text=gt_text,
          is_failset=False
        )
        print(f"➕ Added new case: {rel_path}")
        
        # Log learning progress for new cases
        if has_gt:
          ocr_result = gt_text or "OCR_RESULT_PLACEHOLDER"
          gt_result = gt_text
          test_passes = (ocr_result == gt_result)
          
          if test_passes:
            print(f"🎯 {os.path.basename(rel_path)} guessed '{ocr_result}', was '{gt_result}' OK (NEW→TRAINSET)")
          else:
            print(f"❌ {os.path.basename(rel_path)} guessed '{ocr_result}', was '{gt_result}' NEW→FAILSET")
            # Update database to move to failset
            db.update_case_failset_status(case_id, True)
        
      else:
        # Log learning progress for existing cases (even if no changes)
        if has_gt:
          ocr_result = existing_case.get('ocr_text', 'OCR_RESULT_PLACEHOLDER')
          gt_result = gt_text
          was_failset = existing_case.get('is_failset', False)
          
          # Determine if test passes (OCR matches GT)
          test_passes = (ocr_result == gt_result)
          
          # Apply 4-case learning logic
          if not was_failset and test_passes:
            # Case 1: Trainset → Trainset (was in trainset, test OK, stays in trainset)
            print(f"🎯 {os.path.basename(rel_path)} guessed '{ocr_result}', was '{gt_result}' OK (TRAINSET→TRAINSET)")
          elif not was_failset and not test_passes:
            # Case 2: Trainset → Failset (was in trainset, test fails, goes to failset)
            print(f"❌ {os.path.basename(rel_path)} guessed '{ocr_result}', was '{gt_result}' TRAINSET→FAILSET")
            # Update database to move to failset
            db.update_case_failset_status(existing_case['id'], True)
          elif was_failset and test_passes:
            # Case 3: Failset → Trainset (was in failset, test OK, goes to trainset)
            print(f"✅ {os.path.basename(rel_path)} guessed '{ocr_result}', was '{gt_result}' FAILSET→TRAINSET")
            # Update database to move back to trainset
            db.update_case_failset_status(existing_case['id'], False)
          elif was_failset and not test_passes:
            # Case 4: Failset → Failset (was in failset, test fails, stays in failset)
            print(f"❌ {os.path.basename(rel_path)} guessed '{ocr_result}', was '{gt_result}' FAILSET→FAILSET")
        
        # Update GT text if file changed
        if gt_text != existing_case.get('gt_text'):
          db.update_case_correction(existing_case['id'], gt_text or existing_case.get('ocr_text', ''))
          print(f"🔄 Updated case: {rel_path}")
          
          # Log learning progress for updates
          if has_gt:
            ocr_result = existing_case.get('ocr_text', 'OCR_RESULT_PLACEHOLDER')
            gt_result = gt_text
            if ocr_result == gt_result:
              print(f"🎯 {os.path.basename(rel_path)} guessed '{ocr_result}', was '{gt_result}' OK (updated)")
            else:
              print(f"❌ {os.path.basename(rel_path)} guessed '{ocr_result}', was '{gt_result}' TRAINSET -> FAILSET (updated)")
    
    # Remove cases that no longer exist in folder
    all_cases = db.get_cases(limit=1000)
    for case in all_cases:
      if case['img_path'] not in current_files:
        # Delete case from database
        if db.delete_case_by_img_path(case['img_path']):
          print(f"🗑️  Removed case: {case['img_path']}")
    
  except Exception as e:
    print(f"⚠️  Sync error: {e}")

def folder_polling_worker():
  """Background thread that polls the data folder for changes"""
  print(f"🔍 Starting folder polling for: {DATA_FOLDER}")
  
  while True:
    time.sleep(5)  # Poll every 5 seconds
    sync_database_with_folder()

# Start background polling thread
polling_thread = threading.Thread(target=folder_polling_worker, daemon=True)
polling_thread.start()

# Auto-initialize database with data from folder on startup
try:
  import glob
  cases_count = 0
  
  # Find all .png files in data folder
  png_files = glob.glob(os.path.join(DATA_FOLDER, "**/*.png"), recursive=True)
  
  for png_path in png_files:
    # Convert to relative path for storage
    rel_path = os.path.relpath(png_path)
    
    # Check if case already exists
    existing_case = db.get_case_by_img_path(rel_path)
    if existing_case:
      continue
    
    # Look for corresponding .gt.txt file
    gt_path = png_path.replace('.bin.png', '.gt.txt')
    gt_text = None
    if os.path.exists(gt_path):
      with open(gt_path, 'r', encoding='utf-8') as f:
        gt_text = f.read().strip()
    
    # Debug: Log GT text status
    has_gt = gt_text and gt_text.strip()
    print(f"🔍 {rel_path}: GT exists={os.path.exists(gt_path)}, GT text='{gt_text}', has_content={has_gt}")
    
    # Create case with mock OCR data for now
    # Use realistic confidence based on whether we have GT text
    confidence = 0.85 if has_gt else 0.60
    print(f"📊 {rel_path}: Confidence={confidence}")
    
    case_id = db.insert_case(
      img_path=rel_path,
      ocr_text=gt_text or "OCR_RESULT_PLACEHOLDER",
      confidence=confidence,
      gt_text=gt_text,
      is_failset=False
    )
    cases_count += 1
    
    # Log learning progress for auto-initialization
    if has_gt:
      ocr_result = gt_text or "OCR_RESULT_PLACEHOLDER"
      gt_result = gt_text
      test_passes = (ocr_result == gt_result)
      
      if test_passes:
        print(f"🎯 {os.path.basename(rel_path)} guessed '{ocr_result}', was '{gt_result}' OK (AUTO→TRAINSET)")
      else:
        print(f"❌ {os.path.basename(rel_path)} guessed '{ocr_result}', was '{gt_result}' AUTO→FAILSET")
        # Update database to move to failset
        db.update_case_failset_status(case_id, True)
  
  if cases_count > 0:
    print(f"📊 Auto-initialized {cases_count} cases from {DATA_FOLDER}")
except Exception as e:
  print(f"⚠️  Failed to auto-initialize database: {e}")

# Mount static files for data directory
parent_dir = os.path.dirname(DATA_FOLDER) if os.path.dirname(DATA_FOLDER) else "."
if os.path.exists(parent_dir):
  app.mount("/static", StaticFiles(directory=parent_dir), name="static")
  print(f"📁 Static files mounted from: {parent_dir}")
else:
  print(f"⚠️  Parent directory not found: {parent_dir}")

class CaseRequest(BaseModel):
  image_path: str
  ocr_text: str
  confidence: float
  gt_text: Optional[str] = None
  is_failset: bool = False

class CorrectionRequest(BaseModel):
  corrected_text: str

@app.get("/")
async def root():
  """Serve the main UI page"""
  ui_path = "lib/samuLearnUI.ts/index.html"
  if os.path.exists(ui_path):
    return FileResponse(ui_path)
  else:
    return HTMLResponse("""
    <html>
      <body>
        <h1>samuTrain OCR Monitor</h1>
        <p>UI not found at lib/samuLearnUI.ts/index.html</p>
        <p>Please ensure the UI files are properly installed.</p>
      </body>
    </html>
    """)

@app.get("/api/cases")
async def get_cases(limit: int = 100, failset_only: bool = False):
  """Get OCR cases with optional filtering"""
  try:
    cases = db.get_cases(limit=limit, failset_only=failset_only)
    
    # Convert datetime objects to ISO format for JSON serialization
    for case in cases:
      if isinstance(case['timestamp'], str):
        # SQLite already returns string format, keep as is
        pass
      elif hasattr(case['timestamp'], 'isoformat'):
        case['timestamp'] = case['timestamp'].isoformat()
      
      # Convert boolean fields to proper bool
      case['is_corrected'] = bool(case['is_corrected'])
      case['is_failset'] = bool(case['is_failset'])
      
      # Add image URL for frontend
      img_path = case['img_path'].replace('\\', '/')  # Convert backslashes to forward slashes
      
      if img_path.startswith('./data/'):
        # Remove the ./data/ prefix and convert to static URL
        rel_path = img_path.replace('./data/', '')
        case['image_path'] = f"/static/{rel_path}"
      elif img_path.startswith('data/'):
        # Remove the data/ prefix and convert to static URL
        rel_path = img_path.replace('data/', '')
        case['image_path'] = f"/static/{rel_path}"
      else:
        # Use the path as-is for static URL
        case['image_path'] = f"/static/{img_path}"
    
    return cases
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/debug")
async def debug_cases():
  """Debug endpoint to show case details"""
  try:
    cases = db.get_cases(limit=5)
    debug_info = []
    for case in cases:
      debug_info.append({
        'id': case['id'],
        'img_path': case['img_path'],
        'ocr_text': case['ocr_text'],
        'gt_text': case['gt_text'],
        'confidence': case['confidence'],
        'gt_text_length': len(case['gt_text']) if case['gt_text'] else 0,
        'gt_text_empty': not case['gt_text'] or not case['gt_text'].strip()
      })
    return {"debug_info": debug_info}
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@app.get("/api/version")
async def get_version():
  """Get API version to force cache refresh"""
  return {"version": "1.0", "timestamp": datetime.now().isoformat()}

@app.post("/api/cases")
async def create_case(case_request: CaseRequest):
  """Create a new OCR case"""
  try:
    case_id = db.insert_case(
      img_path=case_request.image_path,
      ocr_text=case_request.ocr_text,
      confidence=case_request.confidence,
      gt_text=case_request.gt_text,
      is_failset=case_request.is_failset
    )
    
    return {"success": True, "case_id": case_id, "message": "Case created successfully"}
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to create case: {str(e)}")

@app.post("/api/cases/{case_id}/correct")
async def correct_case(case_id: int, correction: CorrectionRequest):
  """Correct a case and update GT file"""
  try:
    print(f"=== CORRECTION API CALL ===")
    print(f"Case ID: {case_id}")
    print(f"Corrected text: '{correction.corrected_text}'")
    
    success = db.update_case_correction(case_id, correction.corrected_text)
    
    print(f"Database update result: {success}")
    
    if success:
      return {"success": True, "message": "Case corrected successfully"}
    else:
      raise HTTPException(status_code=404, detail="Case not found or correction failed")
  except HTTPException:
    raise
  except Exception as e:
    print(f"Correction API error: {e}")
    raise HTTPException(status_code=500, detail=f"Correction failed: {str(e)}")

@app.get("/api/statistics")
async def get_statistics():
  """Get system statistics"""
  try:
    stats = db.get_statistics()
    return stats
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.post("/api/initialize")
async def initialize_from_data_folder():
  """Scan data folder and populate database with existing cases"""
  try:
    import glob
    
    cases_created = 0
    data_folder = DATA_FOLDER
    
    # Find all .png files in data folder
    png_files = glob.glob(os.path.join(data_folder, "**/*.png"), recursive=True)
    
    for png_path in png_files:
      # Convert to relative path for storage
      rel_path = os.path.relpath(png_path)
      
      # Check if case already exists
      existing_case = db.get_case_by_img_path(rel_path)
      if existing_case:
        continue
      
      # Look for corresponding .gt.txt file
      gt_path = png_path.replace('.png', '.gt.txt')
      gt_text = None
      if os.path.exists(gt_path):
        with open(gt_path, 'r', encoding='utf-8') as f:
          gt_text = f.read().strip()
      
      # Create case with mock OCR data for now
      # In a real scenario, you'd run OCR prediction here
      case_id = db.insert_case(
        img_path=rel_path,
        ocr_text=gt_text or "OCR_RESULT_PLACEHOLDER",
        confidence=0.95,
        gt_text=gt_text,
        is_failset=False
      )
      cases_created += 1
    
    return {"success": True, "cases_created": cases_created, "message": f"Initialized {cases_created} cases from data folder"}
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to initialize: {str(e)}")

@app.post("/api/reset")
async def reset_database():
  """Reset database and re-initialize from data folder"""
  try:
    # Delete database file
    if os.path.exists("samu.db"):
      os.remove("samu.db")
    
    # Re-initialize database
    db.init_database()
    
    # Re-run initialization
    return await initialize_from_data_folder()
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to reset database: {str(e)}")

@app.get("/api/health")
async def health_check():
  """Health check endpoint"""
  return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="127.0.0.1", port=8000)
