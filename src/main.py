from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import shutil
import time
import glob
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import Database
from bridge import OCRBridge

app = FastAPI(title="samuTrain - AutoDiscovery Mode")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_FOLDER = os.environ.get('SAMUTRAIN_DATA_FOLDER', 'data')
db = Database()
ocr_bridge = OCRBridge()

def discover_new_files():
  """Background task to find new images and create cases"""
  print(f"👀 Auto-Discovery started on: {DATA_FOLDER}")
  while True:
    try:
      # Keressük az összes bin.png-t
      files = glob.glob(os.path.join(DATA_FOLDER, "*.bin.png"))
      
      # Lekérjük a már bent lévő fájlokat a DB-ből a felesleges körök elkerülésére
      # Feltételezve, hogy a db.get_cases egy listát ad vissza
      existing_cases = db.get_cases(limit=10000)
      known_paths = {c['img_path'] for c in existing_cases}
      
      for img_path in files:
        rel_path = os.path.basename(img_path)
        
        if rel_path not in known_paths:
          print(f"✨ New case discovered: {rel_path}")
          # Azonnali predikció az új fájlra (GT még nincs)
          pred, conf = ocr_bridge.predict(img_path)
          
          # Megnézzük, van-e már hozzá véletlenül .gt.txt a mappában
          gt_path = img_path.replace('.bin.png', '.gt.txt')
          gt_text = None
          if os.path.exists(gt_path):
            with open(gt_path, 'r', encoding='utf-8') as f:
              gt_text = f.read().strip()
          
          # Beszúrás az adatbázisba
          db.insert_case(
            img_path=rel_path,
            ocr_text=pred,
            confidence=conf,
            gt_text=gt_text, # Ha nincs GT, None marad -> ez lesz a "FAILSET" jelölt
            is_failset=False
          )
          known_paths.add(rel_path)
          
    except Exception as e:
      print(f"⚠️ Discovery error: {e}")
      
    time.sleep(10) # 10 másodpercenként nézzük át a mappát

# Elindítjuk a figyelőt egy külön szálon
discovery_thread = threading.Thread(target=discover_new_files, daemon=True)
discovery_thread.start()

@app.get("/api/statistics")
async def get_statistics():
  stats = db.get_statistics()
  return stats if stats else {"total": 0, "failset": 0}

@app.get("/api/cases")
async def get_cases(limit: int = 100, failset_only: bool = False):
  return db.get_cases(limit=limit, failset_only=failset_only)

@app.post("/api/train")
async def trigger_training(request: Request):
  try:
    data = await request.json()
    case_ids = data.get('case_ids', [])
    failset_dir = os.path.join(DATA_FOLDER, f"train_{int(time.time())}")
    os.makedirs(failset_dir, exist_ok=True)
    
    count = 0
    for cid in case_ids:
      case = db.get_case(cid)
      if not case or not case.get('gt_text'): continue
      src = os.path.join(DATA_FOLDER, case['img_path'])
      dst = os.path.join(failset_dir, os.path.basename(src))
      if os.path.exists(src):
        shutil.copy2(src, dst)
        with open(dst.replace('.bin.png', '.gt.txt'), 'w', encoding='utf-8') as f:
          f.write(case['gt_text'])
        count += 1
    
    if count > 0:
      ocr_bridge.train_on_failset(failset_dir)
    
    shutil.rmtree(failset_dir, ignore_errors=True)
    return {"success": True, "count": count}
  except Exception as e:
    return {"success": False, "detail": str(e)}

@app.get("/api/status")
async def get_status():
  return {"bridge": ocr_bridge.get_learner_info(), "model": ocr_bridge.model_path}

@app.post("/api/cases/{case_id}/correct")
async def correct_case(case_id: int, request: Request):
  try:
    data = await request.json()
    corrected_text = data.get('corrected_text', '')
    
    success = db.update_case_correction(case_id, corrected_text)
    if success:
      return {"success": True}
    else:
      return {"success": False, "detail": "Failed to save correction"}
  except Exception as e:
    return {"success": False, "detail": str(e)}

# --- UI KISZOLGÁLÁS FIXÁLÁSA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
  # 1. A főoldal (http://127.0.0.1:8000/) kiszolgálása
  @app.get("/", response_class=HTMLResponse)
  async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

  # 2. A statikus fájlok (JS, CSS) elérése (/static/main.js stb.)
  app.mount("/static", StaticFiles(directory=DATA_FOLDER), name="static")

  # 3. "Mentőöv" útvonal: ha a böngésző frissítéskor eltévedne
  @app.get("/{full_path:path}")
  async def catch_all(full_path: str):
    if not full_path.startswith("api/"):
      return FileResponse(os.path.join(STATIC_DIR, "index.html"))
    raise HTTPException(status_code=404)
else:
  print(f"❌ HIBA: Nem találom a static mappát itt: {STATIC_DIR}")
# ------------------------------

if __name__ == "__main__":
  import uvicorn
  # A discovery szál már fut a háttérben
  uvicorn.run(app, host="0.0.0.0", port=8000)