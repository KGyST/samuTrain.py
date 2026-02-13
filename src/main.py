from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from datetime import datetime

from db import Database

app = FastAPI(title="samuTrain OCR Monitor", version="2.0")

# Get data folder from environment variable or default to "data"
DATA_FOLDER = os.environ.get('SAMUTRAIN_DATA_FOLDER', 'data')

# Initialize database
db = Database()

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
      if case['img_path'].startswith('./data/'):
        case['image_path'] = case['img_path']
      else:
        case['image_path'] = case['img_path']
    
    return cases
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

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

@app.get("/api/health")
async def health_check():
  """Health check endpoint"""
  return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="127.0.0.1", port=8000)
