from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
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
    
    # Create case with mock OCR data for now
    # Use realistic confidence based on whether we have GT text
    confidence = 0.85 if gt_text and gt_text.strip() else 0.60
    
    case_id = db.insert_case(
      img_path=rel_path,
      ocr_text=gt_text or "OCR_RESULT_PLACEHOLDER",
      confidence=confidence,
      gt_text=gt_text,
      is_failset=False
    )
    cases_count += 1
  
  if cases_count > 0:
    print(f"📊 Auto-initialized {cases_count} cases from {DATA_FOLDER}")
except Exception as e:
  print(f"⚠️  Failed to auto-initialize database: {e}")

# Mount static files for data directory
if os.path.exists(DATA_FOLDER):
  app.mount("/static", StaticFiles(directory=DATA_FOLDER), name="static")

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
        # Extract just the filename for static URL
        filename = rel_path.split('/')[-1]
        case['image_path'] = f"/static/{filename}"
      elif img_path.startswith('data/'):
        # Remove the data/ prefix and convert to static URL
        rel_path = img_path.replace('data/', '')
        # Extract just the filename for static URL
        filename = rel_path.split('/')[-1]
        case['image_path'] = f"/static/{filename}"
      else:
        # Extract just the filename for static URL
        filename = img_path.split('/')[-1]
        case['image_path'] = f"/static/{filename}"
    
    return cases
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

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

@app.get("/api/health")
async def health_check():
  """Health check endpoint"""
  return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="127.0.0.1", port=8000)
