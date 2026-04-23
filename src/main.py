from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import time
import glob
import json
import asyncio
import threading
import signal
import tempfile
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import Database, get_random_case, get_all_failset_cases, update_case_failset_status
from bridge import OCRBridge

# Import training functions from try_start_training
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import functions directly to avoid encoding conflicts
import shutil

app = FastAPI(title="samuTrain - AutoDiscovery Mode")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_FOLDER = os.environ.get('SAMUTRAIN_DATA_FOLDER', 'data')
db = Database()
ocr_bridge = OCRBridge(db)

def normalize_data_path(data_path):
    """Normalize data path with glob pattern (from try_start_training)"""
    # If already contains glob pattern or .bin.png, return as-is
    if "*" in data_path or data_path.endswith(".bin.png"):
        return data_path
    
    # If it's a directory, add glob pattern
    if os.path.isdir(data_path):
        if not data_path.endswith(os.sep):
            data_path += os.sep
        return data_path + "*.bin.png"
    
    # If it looks like a directory path (ends with /), add glob pattern
    if data_path.endswith("/"):
        return data_path + "*.bin.png"
    
    # Otherwise, assume it's a glob pattern
    return data_path

def verify_dataset(data_pattern):
    """Verify that dataset exists (from try_start_training)"""
    images = glob.glob(data_pattern)
    if not images:
        print(f"❌ No images found matching: {data_pattern}")
        return False
    
    for img in images:
        gt_file = img.replace(".bin.png", ".gt.txt")
        if not os.path.exists(gt_file):
            print(f"❌ Missing Ground Truth file: {gt_file}")
            return False
    return True

async def discover_and_batch_cases():
  """Async batch discovery - polls data folder and adds new cases en masse"""
  print(f"👀 Async batch discovery started on: {DATA_FOLDER}")
  
  while not shutdown_requested:
    try:
      # 1. Get all files from data folder
      files = glob.glob(os.path.join(DATA_FOLDER, "*.bin.png"))
      file_paths = {os.path.basename(f) for f in files}
      
      # 2. Query existing cases from DB (single query)
      existing_cases = db.get_cases(limit=10000)
      known_paths = {c['img_path'] for c in existing_cases}
      
      # 3. Identify new cases (set difference)
      new_files = file_paths - known_paths
      
      if new_files:
        print(f"✨ Found {len(new_files)} new cases for batch processing")
        
        # 4. Batch OCR predictions for new cases
        new_cases_data = []
        for filename in new_files:
          img_path = os.path.join(DATA_FOLDER, filename)
          
          # Get OCR prediction
          try:
            pred, conf = ocr_bridge.predict(img_path)
          except Exception as e:
            print(f"⚠️ OCR prediction failed for {filename}: {e}")
            pred, conf = "ERROR", 0.0
          
          # Check for existing GT file
          gt_path = img_path.replace('.bin.png', '.gt.txt')
          gt_text = None
          if os.path.exists(gt_path):
            with open(gt_path, 'r', encoding='utf-8') as f:
              gt_text = f.read().strip()
          
          new_cases_data.append({
            'img_path': filename,
            'ocr_text': pred,
            'model_prediction': pred,
            'confidence': conf,
            'gt_text': gt_text,
            'is_corrected': False,
            'is_failset': False
          })
        
        # 5. Batch insert all new cases (single transaction)
        if new_cases_data:
          inserted_count = db.batch_insert_cases(new_cases_data)
          if inserted_count > 0:
            print(f"✅ Batch processed {inserted_count} new cases")
          
      else:
        # No new cases, just wait
        pass
        
    except Exception as e:
      print(f"⚠️ Batch discovery error: {e}")
      
    # Sleep with shutdown check
    for i in range(10):  # Check shutdown every second for 10 seconds
      if shutdown_requested:
        break
      await asyncio.sleep(1)

def active_evaluation_loop():
  """Continuously evaluate random cases and update failset status"""
  evaluation_interval = int(os.environ.get('SAMUTRAIN_EVAL_INTERVAL', '10'))  # Reduce from 30s to 10s
  print(f"🔄 Active evaluation started (interval: {evaluation_interval}s)")
  
  while not shutdown_requested:
    try:
      case = get_random_case(weighted=True)
      if case and case.get('gt_text'):
        # Get full path for prediction: img_path is basename (e.g. 010081.bin.png), file is in DATA_FOLDER
        img_path = os.path.join(DATA_FOLDER, case['img_path'])
        pred, conf = ocr_bridge.predict(img_path)

        # Update database with new prediction
        db.update_case_ocr_result(case['id'], pred, conf)
        
        # Compare prediction with ground truth
        success = (pred.strip() == case['gt_text'].strip())
        
        # Update failset status (this will trigger logging)
        update_case_failset_status(case['id'], not success)
        
        print(f"🔍 Evaluated {case['img_path']}: {'✅ OK' if success else '❌ FAIL'} (conf: {conf:.3f})")
        print(f"   📝 Prediction: '{pred}'")
        print(f"   🎯 Ground Truth: '{case['gt_text']}'")
      else:
        print(f"⏭️  No suitable case for evaluation (need GT text)")
        
    except Exception as e:
      print(f"⚠️ Evaluation error: {e}")
      
    time.sleep(evaluation_interval)

def shutdown_training():
  """Train model on failset cases during shutdown with graceful checkpoint saving"""
  print("\n🔄 Starting shutdown training...")
  
  try:
    failset_cases = get_all_failset_cases()
    if not failset_cases:
      print("ℹ️  No failset cases for training")
      return
      
    print(f"📚 Found {len(failset_cases)} failset cases for training")
    
    # Create temporary training directory
    train_dir = tempfile.mkdtemp(prefix="samutrain_shutdown_")
    
    try:
      # Copy failset cases to training directory
      for case in failset_cases:
        img_src = os.path.join(DATA_FOLDER, case['img_path'])
        img_dst = os.path.join(train_dir, os.path.basename(case['img_path']))
        
        if os.path.exists(img_src):
          shutil.copy2(img_src, img_dst)
          
          # Create corresponding .gt.txt file
          gt_dst = img_dst.replace('.bin.png', '.gt.txt')
          with open(gt_dst, 'w', encoding='utf-8') as f:
            f.write(case['gt_text'])
      
      print(f"🎯 Training on {len(failset_cases)} cases...")
      
      # Use continue learning for better codec extension and network handling
      success = ocr_bridge.continue_learning(
        data_folder=train_dir,
        checkpoint_folder=ocr_bridge.model_dir,
        network=None,  # Auto-detect based on data
        backup=False,  # Don't backup during shutdown
        epochs=1
      )
      
      if success:
        print("✅ Model updated successfully")
        # Reload model after training
        ocr_bridge.learner.reload_model()
        print("🔄 Model reloaded")
      else:
        print("❌ Training failed")
        
    finally:
      # Clean up temporary directory
      shutil.rmtree(train_dir, ignore_errors=True)
      print("🧹 Temporary files cleaned up")
      
  except Exception as e:
    print(f"❌ Shutdown training error: {e}")

# Background threads for better management
discovery_thread = None
evaluation_thread = None
shutdown_requested = False

def start_async_discovery_thread():
  """Start async discovery thread with proper management"""
  import asyncio
  
  def run_async_discovery():
    """Run async discovery in new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
      loop.run_until_complete(discover_and_batch_cases())
    except Exception as e:
      print(f"❌ Async discovery error: {e}")
    finally:
      loop.close()
  
  global discovery_thread
  discovery_thread = threading.Thread(target=run_async_discovery, daemon=True, name="async_discovery")
  discovery_thread.start()
  print("👀 Async discovery thread started")

def start_background_threads():
  """Start background threads with proper management"""
  global discovery_thread, evaluation_thread
  
  # Start async discovery thread
  start_async_discovery_thread()
  
  # Start evaluation thread
  evaluation_thread = threading.Thread(target=active_evaluation_loop, daemon=True, name="evaluation")
  evaluation_thread.start()
  print("🔄 Evaluation thread started")

def stop_background_threads():
  """Stop background threads gracefully"""
  global discovery_thread, evaluation_thread, shutdown_requested
  shutdown_requested = True
  
  print("🛑 Stopping background threads...")
  
  # Threads are daemon, so they should exit when main program exits
  # But we can join them with timeout for cleaner shutdown
  if discovery_thread and discovery_thread.is_alive():
    discovery_thread.join(timeout=5)
    print("✅ Discovery thread stopped")
  
  if evaluation_thread and evaluation_thread.is_alive():
    evaluation_thread.join(timeout=5)
    print("✅ Evaluation thread stopped")

# Set up signal handlers for graceful shutdown (from try_start_training)
def signal_handler(signum, frame=None):
  """Handle Ctrl+C gracefully and save best model (from try_start_training)"""
  print(f"\nReceived signal {signum}. Shutting down server gracefully...")
  
  # Stop background threads first
  stop_background_threads()
  
  # Trigger shutdown training
  shutdown_training()
  
  print("Exiting...")
  sys.exit(0)

# Start background threads when module loads
start_background_threads()

# Set up signal handlers for graceful shutdown (from try_start_training)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.get("/api/statistics")
async def get_statistics():
  stats = db.get_statistics()
  # #region agent log
  _log = stats if stats else {"total": 0, "failset": 0}
  try:
    with open("debug-fe785b.log", "a", encoding="utf-8") as f:
      f.write('{"sessionId":"fe785b","runId":"stats","hypothesisId":"H2","location":"main.py:get_statistics","message":"API stats response","data":{"stats":_log},"timestamp":' + str(int(__import__("time").time() * 1000)) + '}\n')
  except Exception: pass
  # #endregion
  return stats if stats else {"total": 0, "failset": 0}

@app.get("/api/updates")
async def get_updated_cases(since: Optional[str] = None):
  """Get cases with update flags for frontend notifications"""
  try:
    from datetime import datetime
    since_dt = None
    if since:
      try:
        since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
      except:
        pass
    
    updated_cases = db.get_updated_cases(since=since_dt)
    return {"success": True, "cases": updated_cases}
  except Exception as e:
    print(f"❌ Get updates API error: {e}")
    return {"success": False, "detail": str(e)}

@app.post("/api/updates/clear")
async def clear_update_flags(request: Request):
  """Clear update flags after frontend reads them"""
  try:
    data = await request.json()
    case_ids = data.get('case_ids', [])
    
    success = db.clear_update_flags(case_ids)
    return {"success": success}
  except Exception as e:
    print(f"❌ Clear updates API error: {e}")
    return {"success": False, "detail": str(e)}

@app.get("/api/cases")
async def get_cases(limit: int = 100, failset_only: bool = False):
  cases = db.get_cases(limit=limit, failset_only=failset_only)
  parent_data_dir = os.path.dirname(DATA_FOLDER)
  data_sub = os.path.basename(DATA_FOLDER)
  for c in cases:
    ip = c.get("img_path", "")
    if "/" not in ip and data_sub:
      c["img_static_path"] = f"{data_sub}/{ip}"
    else:
      c["img_static_path"] = ip
  # #region agent log
  if cases:
    c = cases[0]
    ip = c.get("img_path", "")
    static_path = c.get("img_static_path", ip)
    resolved = os.path.join(parent_data_dir, static_path)
    try:
      with open("debug-fe785b.log", "a", encoding="utf-8") as f:
        f.write('{"sessionId":"fe785b","runId":"cases","hypothesisId":"H1","location":"main.py:get_cases","message":"First case img_path and confidence","data":{"img_path":ip,"img_static_path":static_path,"confidence":c.get("confidence"),"resolved_exists":os.path.exists(resolved)},"timestamp":' + str(int(__import__("time").time() * 1000)) + '}\n')
    except Exception: pass
  # #endregion
  return cases

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
    
    success = False
    if count > 0:
      print(f"🎯 Starting training with {count} samples from {count} cases")
      success = ocr_bridge.train_on_failset(failset_dir)
      if success:
        print("✅ Training successful, model updated")
      else:
        print("❌ Training failed")
    
    # Clean up temporary training directory
    shutil.rmtree(failset_dir, ignore_errors=True)
    return {"success": success, "count": count}
  except Exception as e:
    print(f"❌ Training API error: {e}")
    return {"success": False, "detail": str(e)}

@app.get("/api/status")
async def get_status():
  return {"bridge": ocr_bridge.get_learner_info(), "model": ocr_bridge.model_path}

@app.post("/api/train_new")
async def train_new_model(request: Request):
  """Create and train a new model from scratch"""
  try:
    data = await request.json()
    data_folder = data.get('data_folder')
    model_folder = data.get('model_folder')
    epochs = data.get('epochs', 5)
    network = data.get('network', 'cnn=8:3x3,pool=2x2,lstm=32')
    force = data.get('force', False)
    auto_continue = data.get('auto_continue', False)
    
    if not data_folder:
      return {"success": False, "detail": "data_folder is required"}
    
    if not model_folder:
      return {"success": False, "detail": "model_folder is required"}
    
    # Normalize data path
    data_path = normalize_data_path(data_folder)
    
    # Verify dataset
    if not verify_dataset(data_path):
      return {"success": False, "detail": f"Dataset validation failed for: {data_folder}"}
    
    # Check model folder existence
    model_exists = os.path.exists(model_folder)
    
    # Validate arguments
    if model_exists and not force:
      return {"success": False, "detail": f"Model folder already exists: {model_folder}. Use force=true to overwrite."}
    
    if model_exists and force:
      try:
        shutil.rmtree(model_folder)
        print(f"🗑️ Removed existing model folder: {model_folder}")
      except Exception as e:
        return {"success": False, "detail": f"Failed to remove existing model folder: {e}"}
    
    print(f"🆕 Creating new model: {model_folder}")
    print(f"📊 Data: {data_path}")
    print(f"🎯 Epochs: {epochs}")
    print(f"🌐 Network: {network}")
    
    # Start initial training using bridge
    try:
      success = ocr_bridge.start_new_model(
        data_folder=data_path,
        model_folder=model_folder,
        epochs=epochs,
        network=network
      )
      
      if not success:
        return {"success": False, "detail": "Initial training failed"}
      
      # Handle auto-continuation for new models
      if auto_continue:
        print("\n🤖 Auto-continuation enabled, proceeding...")
        continue_success = ocr_bridge.continue_learning(
          data_folder=data_folder,
          checkpoint_folder=model_folder,
          network=network,
          backup=True,
          epochs=epochs
        )
        
        if continue_success:
          print(f"\n🎉 Complete training workflow finished! Model: {model_folder}")
          return {
            "success": True, 
            "message": "New model created and auto-continued",
            "model_folder": model_folder,
            "epochs": epochs,
            "network": network
          }
        else:
          return {"success": False, "detail": "Initial training succeeded but continuation failed"}
      else:
        print(f"\n🎉 New model created! Model saved to: {model_folder}")
        return {
          "success": True, 
          "message": "New model created successfully",
          "model_folder": model_folder,
          "epochs": epochs,
          "network": network
        }
        
    except Exception as e:
      print(f"❌ New model training error: {e}")
      return {"success": False, "detail": str(e)}
      
  except Exception as e:
    print(f"❌ Train new model API error: {e}")
    return {"success": False, "detail": str(e)}

@app.post("/api/continue_learning")
async def continue_learning(request: Request):
  """Continue learning with new data using the consolidated script"""
  try:
    data = await request.json()
    data_folder = data.get('data_folder')
    checkpoint_folder = data.get('checkpoint_folder', ocr_bridge.model_dir)
    network = data.get('network')
    backup = data.get('backup', True)
    force = data.get('force', False)  # New parameter from try_start_training
    epochs = data.get('epochs')
    
    if not data_folder:
      return {"success": False, "detail": "data_folder is required"}
    
    if not os.path.exists(data_folder):
      return {"success": False, "detail": f"Data folder not found: {data_folder}"}
    
    if not os.path.exists(checkpoint_folder):
      return {"success": False, "detail": f"Checkpoint folder not found: {checkpoint_folder}"}
    
    success = ocr_bridge.continue_learning(
      data_folder=data_folder,
      checkpoint_folder=checkpoint_folder,
      network=network,
      backup=backup,
      force=force,  # Pass force parameter
      epochs=epochs
    )
    
    return {"success": success}
  except Exception as e:
    print(f"❌ Continue learning API error: {e}")
    return {"success": False, "detail": str(e)}

@app.post("/api/model/reload")
async def reload_model():
  """Force reload the model - useful after manual training"""
  try:
    ocr_bridge.learner.reload_model()
    return {"success": True, "message": "Model reloaded successfully"}
  except Exception as e:
    return {"success": False, "detail": str(e)}

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

@app.post("/api/demo/learning")
async def run_learning_demo(request: Request):
  """Run the learning demonstration script"""
  try:
    import subprocess
    import sys
    
    data = await request.json()
    data_folder = data.get('data_folder', 'data/64_case')
    model_folder = data.get('model_folder', 'models/generic_ocr_model_3')
    epochs = data.get('epochs', 1)
    
    # Get script path
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'scripts', 'try_learning_demo.py')
    
    if not os.path.exists(script_path):
      return {"success": False, "detail": "Demo script not found"}
    
    # Prepare command
    cmd = [
      sys.executable, script_path, 
      data_folder, model_folder, 
      '--epochs', str(epochs)
    ]
    
    print(f"🚀 Starting learning demo: {' '.join(cmd)}")
    
    # Run the demo script
    try:
      result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        timeout=300,  # 5 minute timeout
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      )
      
      print(f"Demo completed with return code: {result.returncode}")
      if result.stdout:
        print(f"Demo stdout: {result.stdout[:500]}...")
      if result.stderr:
        print(f"Demo stderr: {result.stderr[:500]}...")
      
      if result.returncode == 0:
        return {
          "success": True, 
          "message": "Learning demo completed successfully",
          "stdout": result.stdout,
          "stderr": result.stderr
        }
      else:
        return {
          "success": False, 
          "detail": f"Demo failed with return code {result.returncode}",
          "stdout": result.stdout,
          "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
      return {"success": False, "detail": "Demo timed out after 5 minutes"}
    except Exception as e:
      return {"success": False, "detail": f"Failed to run demo: {str(e)}"}
      
  except Exception as e:
    print(f"❌ Demo API error: {e}")
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
  # Mount the parent data directory to serve all subfolders (64_case, new_case, single_case, etc.)
  parent_data_dir = os.path.dirname(DATA_FOLDER)  # Get 'data' from 'data/single_case'
  app.mount("/static", StaticFiles(directory=parent_data_dir), name="static")

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