# What is this file for: Unified training script that starts from scratch or continues existing models, with checkpoint validation and proper file handling for continuation training. Consolidated from try_continue_learning.py with all functions in one module, simplified logic, and fixed checkpoint copying to extract files to project root instead of creating/overwriting fake checkpoints. Uses last valid checkpoint with enhanced validation.
# When it is created: 2026-03-17 (complete refactoring)

import os
import sys
import json
import subprocess
import argparse
import glob
import shutil
import signal
from datetime import datetime
from typing import Dict, List, Optional, Set

# Constants (meaningful literals only)
UTF_8 = 'utf-8'
TRAINER_PARAMS = "trainer_params.json"
BEST_CKPT = "best.ckpt.json"
BIN_PNG = ".bin.png"
GT_TXT = ".gt.txt"

# Add src/ and project root to sys.path
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.dirname(scriptDir)
srcDir = os.path.join(projectRoot, 'src')
sys.path.insert(0, srcDir)
sys.path.insert(0, projectRoot)

# Force minimal logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Set UTF-8 encoding for stdout
if sys.platform == "win32":
  import codecs
  sys.stdout = codecs.getwriter(UTF_8)(sys.stdout.detach())

# Global variables for graceful shutdown
trainingProcess: Optional[subprocess.Popen] = None

def normalize_data_path(data_path: str) -> str:
  """Normalize data path to include glob pattern if it's a directory"""
  if "*" in data_path or data_path.endswith(BIN_PNG):
    return data_path
  
  if os.path.isdir(data_path):
    if not data_path.endswith(os.sep):
      data_path += os.sep
    return data_path + "*" + BIN_PNG
  
  if data_path.endswith("/"):
    return data_path + "*" + BIN_PNG
  
  return data_path

def collect_chars(data_folder: str) -> List[str]:
  """Collect all unique characters from ground truth files"""
  assert os.path.exists(data_folder), f"Data folder missing: {data_folder}"
  
  setChars: Set[str] = set()
  for sRoot, _, listFiles in os.walk(s_data_folder):
    for sFile in listFiles:
      if sFile.endswith(GT_TXT):
        sGtFile = os.path.join(sRoot, sFile)
        try:
          with open(sGtFile, 'r', encoding=UTF_8) as f:
            setChars.update(f.read())
        except Exception:
          pass
  return sorted(list(setChars))

def backup_model(model_dir: str) -> Optional[str]:
  """Backup model directory by renaming to .old format"""
  assert os.path.isabs(model_dir), f"Path must be absolute: {model_dir}"
  assert os.path.isdir(model_dir), f"Model directory missing: {model_dir}"

  sTimestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  sBackupDir = f"{model_dir}.{sTimestamp}.old"

  assert not os.path.exists(sBackupDir), f"Backup directory already exists: {sBackupDir}"
  
  try:
    shutil.move(s_model_dir, sBackupDir)
    return sBackupDir
  except Exception:
    return None

def get_last_valid_checkpoint(model_folder: str) -> Optional[str]:
  """Get the last checkpoint folder that has all required files"""
  assert os.path.isabs(model_folder), f"Path must be absolute: {model_folder}"
  assert os.path.isdir(model_folder), f"Model directory missing: {model_folder}"
  
  sCkptRoot = os.path.join(model_folder, "checkpoint")
  if not os.path.exists(sCkptRoot):
    return None

  listCkptFolders = glob.glob(os.path.join(sCkptRoot, "checkpoint_*"))
  if not listCkptFolders:
    return None

  listCkptFolders.sort(key=os.path.getmtime, reverse=True)

  for sCkptFolder in listCkptFolders:
    sParamsPath = os.path.join(sCkptFolder, TRAINER_PARAMS)
    if os.path.exists(sParamsPath) and os.path.isfile(sParamsPath):
      return sCkptFolder

  return None

def get_current_network(model_dir: str) -> Optional[str]:
  """Extract current network architecture from trainer params"""
  assert os.path.isdir(model_dir), f"Model directory missing: {model_dir}"

  try:
    with open(os.path.join(model_dir, TRAINER_PARAMS), 'r') as f:
      dictParams = json.load(f)
    if sNetwork := dictParams.get("network"):
      return sNetwork
  except Exception:
    pass

  sLastCkpt = get_last_valid_checkpoint(model_dir)
  if sLastCkpt:
    try:
      with open(os.path.join(sLastCkpt, TRAINER_PARAMS), 'r') as f:
        dictParams = json.load(f)
      if sNetwork := dictParams.get("network"):
        return sNetwork
    except Exception:
      pass

  return None

def signal_handler(signum: int, _frame: Optional[object] = None) -> None:
  """Handle Ctrl+C gracefully and save best model"""
  global trainingProcess
  if trainingProcess:
    try:
      trainingProcess.send_signal(signal.SIGUSR1)
    except (AttributeError, OSError):
      pass
    
    trainingProcess.terminate()
    try:
      trainingProcess.wait(timeout=60)
    except subprocess.TimeoutExpired:
      trainingProcess.kill()
      trainingProcess.wait()
  
  sys.exit(0)

def verify_dataset(data_pattern: str) -> bool:
  """Verify that dataset exists and contains ground truth files"""
  listImages = glob.glob(data_pattern)
  if not listImages:
    return False
  
  for sImg in listImages:
    sGtFile = sImg.replace(BIN_PNG, GT_TXT)
    if not os.path.exists(sGtFile):
      return False
  return True

def extract_checkpoint_files(model_dir: str, target_dir: str) -> bool:
  """Extract essential checkpoint files to target directory"""
  assert os.path.isdir(target_dir), f"Target directory missing: {target_dir}"
  
  sLastCkpt = get_last_valid_checkpoint(model_dir)
  if sLastCkpt is None:
    return False
  
  bExtracted = False
  for sFile in [BEST_CKPT, TRAINER_PARAMS]:
    sDst = os.path.join(target_dir, sFile)
    sSrc = None
    if os.path.exists(sTry := os.path.join(sLastCkpt, sFile)):
      sSrc = sTry
    elif os.path.exists(sTry := os.path.join(model_dir, sFile)):
      sSrc = sTry

    if sSrc:
      shutil.copy2(sSrc, sDst)
      bExtracted = True
  
  sBestCkptSrc = os.path.join(s_model_dir, "best.ckpt")
  sBestCkptDst = os.path.join(s_target_dir, "best.ckpt")
  if os.path.exists(sBestCkptSrc):
    if os.path.exists(sBestCkptDst):
      shutil.rmtree(sBestCkptDst)
    shutil.copytree(sBestCkptSrc, sBestCkptDst)
    bExtracted = True
  
  return bExtracted

def start_initial_training(data_pattern: str, epochs: int, output_dir: str, network: Optional[str]) -> bool:
  """Start training from scratch using train.py logic"""
  global trainingProcess
  if not network:
    network = "cnn=8:3x3,pool=2x2,lstm=32"
  
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  sAbsDataPattern = os.path.abspath(data_pattern)
  sAbsOutputDir = os.path.abspath(output_dir)
  
  listCmd = [
    sys.executable, "-m", "calamari_ocr.scripts.train",
    "--train.images", sAbsDataPattern,
    "--trainer.epochs", str(epochs),
    "--trainer.output_dir", sAbsOutputDir,
    "--trainer.best_model_prefix", "best",
    "--trainer.gen", "TrainOnly",
    "--train.batch_size", "1",
    "--val.batch_size", "1",
    "--codec.auto_compute", "True",
    "--network", network
  ]
  
  try:
    trainingProcess = subprocess.Popen(listCmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    if trainingProcess.stdout:
      for sLine in iter(trainingProcess.stdout.readline, ''):
        if sLine.strip():
          pass # Minimal output
    
    trainingProcess.wait()
    return trainingProcess.returncode == 0
      
  except KeyboardInterrupt:
    signal_handler(signal.SIGINT)
    return False
  except Exception:
    return False
  finally:
    trainingProcess = None

def continue_learning(model_dir: str, continue_data: str, network: Optional[str] = None, backup: bool = True) -> bool:
  """Continue learning using library API with backup and file handling"""
  assert os.path.isdir(model_dir), f"Model directory missing: {model_dir}"

  sBackupDir = None
  if backup:
    sBackupDir = backup_model(model_dir)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  if sBackupDir and extract_checkpoint_files(sBackupDir, model_dir):
    pass
  elif not sBackupDir:
    sLastCkpt = get_last_valid_checkpoint(model_dir)
    if not (sLastCkpt and extract_checkpoint_files(sLastCkpt, model_dir)):
      return False
  else:
    return False

  if network is None:
    network = get_current_network(model_dir)

  assert network, "Network architecture not found"

  sDataPattern = normalize_data_path(continue_data)
  if not verify_dataset(sDataPattern):
    return False

  listChars = collect_chars(continue_data)

  sCharsetFile = None
  if listChars:
    sCharsetFile = os.path.join(model_dir, "extended_charset.txt")
    with open(sCharsetFile, 'w', encoding=UTF_8) as f:
      f.write(''.join(listChars))

  try:
    from calamari_ocr.ocr.scenario import CalamariScenario
    from calamari_ocr.scripts.train import main as calamari_train
    
    trainerParams = CalamariScenario.default_trainer_params()
    trainerParams.output_dir = model_dir
    trainerParams.epochs = 100
    trainerParams.auto_upgrade_checkpoints = True
    trainerParams.network = network
    trainerParams.early_stopping.n_to_go = -1
    trainerParams.gen.train.images = [sDataPattern]
    trainerParams.gen.train.skip_invalid = True
    trainerParams.gen.setup.train.batch_size = 1
    trainerParams.gen.setup.train.num_processes = 1
    trainerParams.gen.val.images = trainerParams.gen.train.images
    trainerParams.gen.setup.val.num_processes = 1
    
    sCheckpointFile = os.path.join(model_dir, "best.ckpt")
    if not os.path.exists(sCheckpointFile):
      sCheckpointFile = os.path.join(model_dir, TRAINER_PARAMS)
    
    if not os.path.exists(sCheckpointFile):
      return False
    
    trainerParams.warmstart.model = sCheckpointFile
    trainerParams.warmstart.allow_partial = True
    trainerParams.warmstart.trim_graph_name = False
    
    if sCharsetFile:
      trainerParams.codec.include_files = [sCharsetFile]
      trainerParams.codec.auto_compute = True
      trainerParams.codec.keep_loaded = True
    
    trainerParams.progress_bar = False
    calamari_train(trainerParams)
    return True

  except KeyboardInterrupt:
    signal_handler(signal.SIGINT)
    return False
  except Exception:
    return False

def main() -> None:
  """Main entry point for training script"""
  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)
  
  argParser = argparse.ArgumentParser(description="Train or continue OCR model")
  argParser.add_argument("data_folder", help="Path to training data folder")
  argParser.add_argument("model_folder", help="Path to model folder")
  argParser.add_argument("--new", action="store_true", help="Create new model")
  argParser.add_argument("--epochs", type=int, default=5, help="Training epochs")
  argParser.add_argument("--network", default=None, help="Network architecture")
  argParser.add_argument("--auto-continue", action="store_true", help="Auto-continue training")
  argParser.add_argument("--force", action="store_true", help="Force overwrite")
  
  parsedArgs = argParser.parse_args()
  sDataPath = normalize_data_path(parsedArgs.data_folder)
  
  if not verify_dataset(sDataPath):
    sys.exit(1)
  
  bModelExists = os.path.exists(parsedArgs.model_folder)
  
  if parsedArgs.new and bModelExists:
    if not parsedArgs.force:
      sys.exit(1)
    shutil.rmtree(parsedArgs.model_folder)
    bModelExists = False
  
  if not parsedArgs.new and not bModelExists:
    sys.exit(1)
  
  if parsedArgs.new:
    if not start_initial_training(sDataPath, parsedArgs.epochs, parsedArgs.model_folder, parsedArgs.network):
      sys.exit(1)
    
    if parsedArgs.auto_continue:
      if not continue_learning(parsedArgs.model_folder, sDataPath, parsedArgs.network):
        sys.exit(1)
  else:
    if not continue_learning(parsedArgs.model_folder, parsedArgs.data_folder, parsedArgs.network):
      sys.exit(1)

if __name__ == "__main__":
  main()
