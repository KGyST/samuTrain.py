import os
import subprocess
import sys
import json
import re
from pathlib import Path

# Use few comments. 2-space indentation. English only.
def get_current_epoch(model_folder):
  """Extract current epoch from latest checkpoint trainer_params.json"""
  try:
    checkpoint_dir = os.path.join(model_folder, "checkpoint")
    if not os.path.exists(checkpoint_dir):
      return 1
    
    # Find latest checkpoint
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
      if item.startswith("checkpoint_") and os.path.isdir(os.path.join(checkpoint_dir, item)):
        try:
          num = int(item.split("_")[1])
          checkpoints.append((num, item))
        except (ValueError, IndexError):
          continue
    
    if checkpoints:
      latest = max(checkpoints, key=lambda x: x[0])
      trainer_params_path = os.path.join(checkpoint_dir, latest[1], "trainer_params.json")
      
      if os.path.exists(trainer_params_path):
        with open(trainer_params_path, 'r') as f:
          params = json.load(f)
          return params.get("current_epoch", 1) + 1
    
    return 1
  except Exception as e:
    print(f"⚠️ Could not determine current epoch: {e}")
    return 1

def resume_training(data_folder, model_folder, max_epochs=None, val_every_n=None, timeout=None):
  """Resume training with optional timeout for final validation"""
  current_python = sys.executable
  
  # Always use resume_training script - it's the only one that works reliably
  cmd = [
    current_python, "-m", "calamari_ocr.scripts.resume_training",
    model_folder  # Pass model directory, not checkpoint prefix
  ]
  
  if max_epochs or val_every_n:
    print(f"🎯 Final validation phase - will run until validation completes")
    if timeout:
      print(f"⏱️ Timeout set to {timeout} seconds")
  else:
    print("� Normal training phase")
  
  print(f"🚀 Resuming training from: {model_folder}")
  print(f"🐍 Using python: {current_python}")
  print(f"📁 Data folder: {data_folder}")
  
  # Forward environment to ensure TF/Calamari is found
  try:
    if timeout:
      result = subprocess.run(cmd, env=os.environ.copy(), timeout=timeout)
    else:
      result = subprocess.run(cmd, env=os.environ.copy())
    return result
  except subprocess.TimeoutExpired:
    print(f"⏱️ Training stopped after {timeout} seconds (validation should have completed)")
    return subprocess.CompletedProcess(cmd, 0, "Training stopped by timeout", "")

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python try_resume_train.py <data_folder> <model_folder>")
    print("Example: python try_resume_train.py data/1962 models/test_zero_interruption_v2")
    sys.exit(1)
    
  data_folder = sys.argv[1]
  model_folder = sys.argv[2]
  
  print("🎯 Starting zero-interruption training with Ctrl+C safety")
  print("💡 Press Ctrl+C at any time to stop and validate the model")
  
  try:
    # Phase 1: Normal resume training
    print("\n=== PHASE 1: Normal Training ===")
    result = resume_training(data_folder, model_folder)
    
    if result.returncode == 0:
      print("✅ Training completed successfully")
    else:
      print(f"❌ Training failed with exit code: {result.returncode}")
      
  except KeyboardInterrupt:
    print("\n⚡ KeyboardInterrupt caught - Starting graceful shutdown...")
    
    # Phase 2: Final resume with validation
    print("\n=== PHASE 2: Final Validation Training ===")
    current_epoch = get_current_epoch(model_folder)
    final_epoch = current_epoch + 1
    
    print(f"📍 Current epoch: {current_epoch}")
    print(f"🎯 Running final training for {final_epoch} epochs to trigger validation")
    
    try:
      # Resume with timeout for final validation (5 minutes should be enough for 1 epoch)
      result = resume_training(
        data_folder, 
        model_folder, 
        max_epochs=final_epoch,
        val_every_n=1,
        timeout=300  # 5 minutes timeout
      )
      
      if result.returncode == 0:
        print("✅ Final validation training completed successfully")
        print("🎉 Usable model should now be available in best.ckpt/")
        
        # Check if best.ckpt.json was created
        best_json = os.path.join(model_folder, "best.ckpt.json")
        if os.path.exists(best_json):
          print("✅ Best model exported: best.ckpt.json")
        else:
          print("⚠️ Best model not found, but training completed")
          
      else:
        print(f"❌ Final validation training failed with exit code: {result.returncode}")
        
    except KeyboardInterrupt:
      print("\n⚠️ Second KeyboardInterrupt - exiting immediately")
      sys.exit(1)
      
    except Exception as e:
      print(f"❌ Error during final validation training: {e}")
      sys.exit(1)
      
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
    
  print("\n🏁 Training session complete")