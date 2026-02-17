from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys
import sqlite3
import threading
import time
import argparse
import json
import shutil
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db import Database
from bridge import OCRBridge

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

# Initialize OCR bridge
ocr_bridge = OCRBridge()
print(f"🔧 OCR Bridge Status: {ocr_bridge.get_learner_info()}")

# Folder polling and synchronization
import threading
import time
from pathlib import Path

# Learning progress counter and auto-training trigger
learning_step_counter = 0
last_training_time = 0
TRAINING_INTERVAL = 30  # seconds between automatic training attempts

def sync_database_with_folder():
  """Synchronize database with actual data folder contents"""
  global learning_step_counter
  learning_step_counter += 1
  
  try:
    import glob
    
    # Only print learning steps every 5 cycles to reduce noise
    if learning_step_counter % 5 == 1:
      print(f"🔄 Learning Step #{learning_step_counter} - Scanning {DATA_FOLDER}")
    
    # Get all image files in data folder (excluding temp directories)
    current_files = set()
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bin.png']:
      for file_path in Path(DATA_FOLDER).rglob(ext):
        # Skip temp directories
        if 'temp_failset' in str(file_path):
          continue
        rel_path = os.path.relpath(file_path, DATA_FOLDER)
        current_files.add(rel_path)
      
    for png_path in current_files:
      # Look for corresponding .gt.txt file
      gt_path = png_path.replace('.bin.png', '.gt.txt')
      gt_text = None
      if os.path.exists(gt_path):
        with open(gt_path, 'r', encoding='utf-8') as f:
          gt_text = f.read().strip()
      
      # Debug: Log GT text status for sync
      has_gt = gt_text and gt_text.strip()
      
      # Check if case exists
      existing_case = db.get_case_by_img_path(rel_path)
      
      if not existing_case:
        # Run real OCR prediction
        full_image_path = os.path.abspath(png_path)  # Use absolute path directly
        ocr_text, confidence = ocr_bridge.predict(full_image_path, gt_text or "")
        
        print(f"📊 {rel_path}: OCR='{ocr_text}', confidence={confidence:.3f}, has_gt={has_gt}")
        case_id = db.insert_case(
          img_path=rel_path,
          ocr_text=ocr_text,
          confidence=confidence,
          gt_text=gt_text,
          is_failset=False
        )
        print(f"➕ Added new case: {rel_path}")
        
        # Log learning progress for new cases
        if has_gt:
          ocr_result = ocr_text
          gt_result = gt_text
          test_passes = (ocr_result == gt_result)
          
          if test_passes:
            print(f"🎯 {os.path.basename(rel_path)} guessed '{ocr_result}', was '{gt_result}' OK (NEW→TRAINSET)")
          else:
            print(f"❌ {os.path.basename(rel_path)} guessed '{ocr_result}', was '{gt_result}' NEW→FAILSET")
            # Update database to move to failset
            db.update_case_failset_status(case_id, True)
        
      else:
        # Run real OCR prediction for existing cases to check learning progress
        # Always update model_prediction with latest OCR result
        full_image_path = os.path.abspath(png_path)  # Use absolute path directly
        ocr_text, confidence = ocr_bridge.predict(full_image_path, gt_text or "")
        
        # Always update the latest model prediction
        db.update_model_prediction(existing_case['id'], ocr_text)
        
        # But skip updating ocr_text/confidence for user-corrected cases to preserve corrections
        if not existing_case.get('is_corrected', False):
          # Check if case has ground truth in database
          case_has_gt = existing_case.get('gt_text') is not None and existing_case.get('gt_text').strip() != ''
          
          if not case_has_gt:
            # Case has no GT - bootstrap by writing OCR result to .txt file as initial GT
            gt_path = png_path.rsplit('.', 1)[0] + '.gt.txt'
            try:
              # Check if GT file already exists and has content (preserve user corrections)
              existing_gt_content = None
              if os.path.exists(gt_path):
                with open(gt_path, 'r', encoding='utf-8') as f:
                  existing_gt_content = f.read().strip()
                
                # If GT file has non-empty content, preserve it (likely user correction)
                if existing_gt_content:
                  print(f"⏭️  Preserving existing GT file content: {os.path.basename(gt_path)} = '{existing_gt_content}'")
                  # Update database with existing content
                  db.update_case_gt_text(existing_case['id'], existing_gt_content)
                  gt_text = existing_gt_content
                  has_gt = True
                  continue  # Skip to next case
              
              # Check if case is user-corrected (shouldn't overwrite user corrections)
              is_corrected = existing_case.get('is_corrected', False)
              
              if is_corrected:
                print(f"⏭️  Skipping bootstrap for {os.path.basename(gt_path)} - case is user-corrected")
              else:
                with open(gt_path, 'w', encoding='utf-8') as f:
                  f.write(ocr_text)
                
                old_display = f"'{existing_gt_content}'" if existing_gt_content else "none"
                print(f"✏️  Bootstrapped GT file: {os.path.basename(gt_path)} | {old_display} → '{ocr_text}'")
                
                # Update database with GT text
                db.update_case_gt_text(existing_case['id'], ocr_text)
                # Update OCR result and confidence
                db.update_case_ocr_result(existing_case['id'], ocr_text, confidence)
                
                # Now treat as having GT for learning
                gt_text = ocr_text
                has_gt = True
            except Exception as e:
              print(f"⚠️  Failed to bootstrap GT file {gt_path}: {e}")
          else:
            # Case has GT - update OCR result only
            db.update_case_ocr_result(existing_case['id'], ocr_text, confidence)
        else:
          # For corrected cases, use the stored OCR result for logging
          ocr_text = existing_case.get('ocr_text', '')
          confidence = existing_case.get('confidence', 0.8)
        
        # Log learning progress for existing cases (even if no changes)
        if has_gt:
          ocr_result = ocr_text
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
        
        # Update database if GT file changed (but NEVER write back to GT files)
        if gt_text != existing_case.get('gt_text'):
          old_gt = existing_case.get('gt_text') or "none"
          new_gt = gt_text or "none"
          
          # Check if case is user-corrected - log differently
          is_corrected = existing_case.get('is_corrected', False)
          
          if is_corrected:
            print(f"🔄 External GT change detected: {rel_path} | '{old_gt}' → '{new_gt}' (user-corrected, restoring user data)")
            # Restore the correct GT text to the file
            gt_path = png_path.rsplit('.', 1)[0] + '.gt.txt'
            try:
              with open(gt_path, 'w', encoding='utf-8') as f:
                f.write(old_gt)  # Write back the correct user correction
              print(f"✅ Restored GT file: {os.path.basename(gt_path)} = '{old_gt}'")
            except Exception as e:
              print(f"⚠️  Failed to restore GT file {gt_path}: {e}")
            # Don't update database for user-corrected cases
          else:
            # Only update database for non-corrected cases
            db.update_case_gt_text(existing_case['id'], gt_text)
            print(f"🔄 Updated database: {rel_path} | '{old_gt}' → '{new_gt}'")
    
    # Remove cases that no longer exist in folder
    all_cases = db.get_cases(limit=1000)
    for case in all_cases:
      if case['img_path'] not in current_files:
        # Delete case from database
        if db.delete_case_by_img_path(case['img_path']):
          print(f"🗑️  Removed case: {case['img_path']}")
    
    # AUTO-TRAINING: Check if we have failset cases and should trigger training
    global last_training_time
    current_time = time.time()
    time_since_last_training = current_time - last_training_time
    
    # Get failset cases
    failset_cases = db.get_cases(limit=1000, failset_only=True)
    
    if failset_cases and len(failset_cases) > 0 and time_since_last_training > TRAINING_INTERVAL:
      print(f"🚀 Auto-training triggered: {len(failset_cases)} failset cases found, {int(time_since_last_training)}s since last training")
      
      # Trigger automatic training
      try:
        # Create temporary directory for failset training data
        failset_dir = os.path.join(DATA_FOLDER, "temp_failset")
        os.makedirs(failset_dir, exist_ok=True)
        
        # Copy failset images and GT files to temp directory
        trained_count = 0
        for case in failset_cases:
          img_path = case['img_path']
          gt_text = case['gt_text']
          
          # Skip if no GT text
          if not gt_text:
            continue
            
          # Get absolute paths - normalize path separators properly
          img_path_normalized = img_path.replace('\\', '/')
          if img_path_normalized.startswith('data/'):
            src_img = img_path_normalized
          else:
            src_img = os.path.join(DATA_FOLDER, img_path_normalized)
          
          dst_img = os.path.join(failset_dir, os.path.basename(src_img))
          
          # Normalize both paths for comparison
          src_img = os.path.normpath(src_img)
          dst_img = os.path.normpath(dst_img)
          
          # Copy image file
          if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            
            # Create GT file
            gt_filename = os.path.basename(src_img).replace('.bin.png', '.gt.txt')
            gt_path = os.path.join(failset_dir, gt_filename)
            
            with open(gt_path, 'w', encoding='utf-8') as f:
              f.write(gt_text)
            
            trained_count += 1
        
        if trained_count > 0:
          print(f"🚀 Starting auto-training on {trained_count} failset cases...")
          
          # Trigger training
          ocr_bridge.train_on_failset(failset_dir)
          
          # Update last training time
          last_training_time = current_time
          print(f"✅ Auto-training completed, next check in {TRAINING_INTERVAL}s")
        
        # Clean up temp directory
        shutil.rmtree(failset_dir, ignore_errors=True)
        
      except Exception as e:
        print(f"⚠️ Auto-training error: {e}")
    
  except Exception as e:
    print(f"⚠️ Sync error: {e}")

def folder_polling_worker():
  """Background thread that polls data folder for changes"""
  while True:
    try:
      sync_database_with_folder()
      time.sleep(2)  # Reduced from 5 seconds to 2 seconds for more frequent learning
    except Exception as e:
      print(f"⚠️  Folder polling error: {e}")
      time.sleep(5)

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
    
    # Create case with real OCR prediction
    full_image_path = os.path.abspath(png_path)  # Use absolute path directly
    ocr_text, confidence = ocr_bridge.predict(full_image_path, gt_text or "")
    
    print(f"📊 {rel_path}: OCR='{ocr_text}', confidence={confidence:.3f}")
    
    case_id = db.insert_case(
      img_path=rel_path,
      ocr_text=ocr_text,
      confidence=confidence,
      gt_text=gt_text,
      is_failset=False,
      model_prediction=ocr_text
    )
    cases_count += 1
    
    # Log learning progress for auto-initialization
    if has_gt:
      ocr_result = ocr_text
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
  ui_path = os.path.join(os.path.dirname(__file__), "..", "lib", "samuLearnUI.ts", "index.html")
  ui_path = os.path.abspath(ui_path)
  
  print(f"🔍 Looking for UI file at: {ui_path}")
  print(f"📁 File exists: {os.path.exists(ui_path)}")
  
  if os.path.exists(ui_path):
    print(f"✅ Serving UI file: {ui_path}")
    return FileResponse(ui_path, media_type="text/html")
  else:
    print(f"❌ UI file not found: {ui_path}")
    return HTMLResponse(f"""
    <html>
      <body>
        <h1>samuTrain OCR Monitor</h1>
        <p>UI not found at {ui_path}</p>
        <p>Current working directory: {os.getcwd()}</p>
        <p>Please ensure the UI files are properly installed.</p>
      </body>
    </html>
    """, status_code=404)

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
      
      # Add image URL for frontend - include data folder for cache busting
      img_path = case['img_path']
      img_path_normalized = img_path.replace('\\', '/')  # Normalize path separators
      
      # Extract the data folder name from DATA_FOLDER for URL generation
      data_folder_name = os.path.basename(DATA_FOLDER.rstrip('/\\'))
      
      # Find the relative path within the data folder
      # The path should be data/folder/filename.png, so we need folder/filename.png
      if f"{data_folder_name}/" in img_path_normalized:
        # Extract everything after the folder name
        rel_path = img_path_normalized.split(f"{data_folder_name}/", 1)[1]
        case['image_path'] = f"/static/{data_folder_name}/{rel_path}"
      else:
        # Fallback - just use the filename with folder prefix
        filename = os.path.basename(img_path_normalized)
        case['image_path'] = f"/static/{data_folder_name}/{filename}"
    
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

    print ("DATA_FOLDER")
    print (DATA_FOLDER)
    
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

@app.post("/api/train")
async def train_on_failset():
  """Trigger training on failset cases"""
  try:
    import glob
    
    # Get failset cases from database
    failset_cases = db.get_cases(limit=1000, failset_only=True)
    
    if not failset_cases:
      return {"success": False, "message": "No failset cases found for training"}
    
    # Create temporary directory for failset training data
    failset_dir = os.path.join(DATA_FOLDER, "temp_failset")
    os.makedirs(failset_dir, exist_ok=True)
    
    # Copy failset images and GT files to temp directory
    trained_count = 0
    for case in failset_cases:
      img_path = case['img_path']
      gt_text = case['gt_text']
      
      print(f"Processing case: img_path='{img_path}', gt_text='{gt_text}'")
      
      # Skip temp directories and ensure we have GT text
      img_path_normalized_for_filter = img_path.replace('\\', '/')
      has_temp = 'temp_failset' in img_path_normalized_for_filter
      print(f"Filter check: normalized='{img_path_normalized_for_filter}', has_temp={has_temp}, has_gt={bool(gt_text)}")
      
      if not gt_text or has_temp:
        print(f"Skipping case: no_gt={not gt_text}, has_temp={has_temp}")
        continue
        
      # Get absolute paths - normalize path separators
      img_path_normalized = img_path.replace('\\', '/')
      if img_path_normalized.startswith('data/'):
        src_img = img_path_normalized
      else:
        src_img = os.path.join(DATA_FOLDER, img_path_normalized).replace('\\', '/')
      
      dst_img = os.path.join(failset_dir, os.path.basename(src_img)).replace('\\', '/')
      
      print(f"File paths: src_img='{src_img}', dst_img='{dst_img}'")
      print(f"Source exists: {os.path.exists(src_img)}")
      
      # Copy image file
      if os.path.exists(src_img):
        shutil.copy2(src_img, dst_img)
        
        # Create GT file
        gt_filename = os.path.basename(src_img).replace('.bin.png', '.gt.txt')
        gt_path = os.path.join(failset_dir, gt_filename)
        
        with open(gt_path, 'w', encoding='utf-8') as f:
          f.write(gt_text)
        
        trained_count += 1
    
    if trained_count == 0:
      return {"success": False, "message": "No valid failset cases with GT text found"}
    
    print(f"🚀 Starting training on {trained_count} failset cases...")
    
    # Trigger training
    ocr_bridge.train_on_failset(failset_dir)
    
    # Clean up temp directory
    shutil.rmtree(failset_dir, ignore_errors=True)
    
    return {"success": True, "message": f"Training completed on {trained_count} failset cases"}
    
  except Exception as e:
    print(f"Training API error: {e}")
    raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="127.0.0.1", port=8000)
