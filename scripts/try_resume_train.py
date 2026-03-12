import os
import subprocess
import sys
import json
import time

# Force minimal logging at the OS level (3 = ERROR only)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

def get_current_epoch(model_folder):
  try:
    ckpt_dir = os.path.join(model_folder, "checkpoint")
    if not os.path.exists(ckpt_dir): return 1
    ckpts = [int(d.split("_")[1]) for d in os.listdir(ckpt_dir) if d.startswith("checkpoint_")]
    return max(ckpts) if ckpts else 1
  except: return 1

def modify_params(model_folder, epochs, val_every_n):
  path = os.path.join(model_folder, "trainer_params.json")
  if os.path.exists(path):
    with open(path, 'r') as f: params = json.load(f)
    params["epochs"] = epochs
    params["val_every_n"] = val_every_n
    # Sync internal epoch counter too
    params["current_epoch"] = get_current_epoch(model_folder)
    with open(path, 'w') as f: json.dump(params, f, indent=2)
    return True
  return False

if __name__ == "__main__":
  model_dir = sys.argv[2] if len(sys.argv) > 2 else "models/test_v2"
  
  # Base command WITHOUT the problematic --v 0
  cmd = [sys.executable, "-m", "calamari_ocr.scripts.resume_training", model_dir]
  
  print("🚀 Phase 1: Training starting (Silent Mode)...")
  try:
    # Use Popen to control the process group
    proc = subprocess.Popen(cmd, env=os.environ.copy(), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    proc.wait()
  except KeyboardInterrupt:
    print("\n🛑 Stop requested. Killing training tree...")
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    time.sleep(2)

  print("\n🔍 Phase 2: Final Validation Pass")
  current = get_current_epoch(model_dir)
  if modify_params(model_dir, current, 1):
    # Running final pass to generate best.ckpt.json
    print(f"🎯 Forcing final validation at epoch {current}...")
    subprocess.run(cmd, env=os.environ.copy())
  else:
    print("❌ Could not patch trainer_params.json")

  best_json = os.path.join(model_dir, "best.ckpt.json")
  if os.path.exists(best_json):
    print(f"✅ Success: {best_json} generated.")
  else:
    print("⚠️ Process finished, but best.ckpt.json is missing.")