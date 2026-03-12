# What is this file for: Library-based continue learning engine for samuTrain V2
# When it is created: 2026-03-10

import os
import sys
import json
import shutil
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from pathlib import Path

# Add project root to sys.path for Calamari imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Force minimal logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'

try:
  from calamari_ocr.ocr.training.params import TrainerParams
  from calamari_ocr.ocr.scenario import CalamariScenario
  from calamari_ocr.ocr.training.trainer import Trainer
  from calamari_ocr.ocr.scenario_params import CalamariScenarioParams
  from calamari_ocr.ocr.dataset.codec import CodecConstructionParams
  from calamari_ocr.scripts.train import main as calamari_train
  LIB_MODE = True
except ImportError:
  LIB_MODE = False
  logging.error("Calamari library not available")
  
  # Create dummy classes for type hints when library not available
  class TrainerParams:
    pass

class ContinueLearningEngine:
  """Library-based continue learning engine with async support and unicode logging"""
  
  def __init__(self, model_dir: str):
    self.model_dir = model_dir
    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.INFO)
    
    # Setup unicode console handler
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    self.logger.addHandler(handler)
    
    if not LIB_MODE:
      self.logger.error("Calamari library not available - continue learning disabled")
    
  def collect_chars(self, data_folder: str) -> list:
    """Collect unique characters from all .gt.txt files in data folder"""
    chars = set()
    data_path = Path(data_folder)
    
    if not data_path.exists():
      raise FileNotFoundError(f"Data folder not found: {data_folder}")
    
    for gt_file in data_path.glob("*.gt.txt"):
      try:
        with open(gt_file, 'r', encoding='utf-8') as f:
          content = f.read()
          chars.update(content)
          self.logger.debug(f"Found {len(content)} chars in {gt_file.name}")
      except Exception as e:
        self.logger.warning(f"Failed to read {gt_file}: {e}")
    
    result = sorted(list(chars))
    self.logger.info(f"Collected {len(result)} unique characters: {self._format_unicode_chars(result)}")
    return result
  
  def _format_unicode_chars(self, chars: list) -> str:
    """Format unicode characters for display with proper escaping"""
    formatted = []
    for char in chars:
      if ord(char) < 32 or ord(char) > 126:
        # Control characters or non-ASCII - show as unicode escape
        formatted.append(f"\\u{ord(char):04x}")
      else:
        formatted.append(char)
    return ''.join(formatted)
  
  def get_current_network(self, checkpoint_folder: str) -> str:
    """Extract current network architecture from trainer params"""
    params_path = os.path.join(checkpoint_folder, "trainer_params.json")
    if not os.path.exists(params_path):
      self.logger.warning(f"No trainer_params.json found, using default network")
      return "cnn=8:3x3,pool=2x2,lstm=32"
    
    try:
      with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
      network = params.get("network", "cnn=8:3x3,pool=2x2,lstm=32")
      self.logger.info(f"Current network: {network}")
      return network
    except Exception as e:
      self.logger.error(f"Failed to read trainer params: {e}")
      return "cnn=8:3x3,pool=2x2,lstm=32"
  
  def backup_model(self, model_dir: str) -> Optional[str]:
    """Backup model directory to .old folder with timestamp"""
    if not os.path.exists(model_dir):
      self.logger.warning(f"Model directory not found: {model_dir}")
      return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{model_dir}_{timestamp}.old"
    
    try:
      shutil.copytree(model_dir, backup_dir)
      self.logger.info(f"Model backed up to: {backup_dir}")
      return backup_dir
    except Exception as e:
      self.logger.error(f"Failed to backup model: {e}")
      return None
  
  def create_trainer_params(self, data_folder: str, checkpoint_folder: str, 
                          network: str, chars: list, new_model_dir: str) -> TrainerParams:
    """Create TrainerParams with all prototype parameters using library API"""
    
    # Get default trainer params
    trainer_params = CalamariScenario.default_trainer_params()
    
    # Set basic parameters
    trainer_params.output_dir = new_model_dir
    trainer_params.epochs = 100
    trainer_params.auto_upgrade_checkpoints = True
    trainer_params.network = network
    
    # Early stopping
    trainer_params.early_stopping.n_to_go = -1  # Disable early stopping
    
    # Training data
    trainer_params.gen.train.images = [os.path.join(data_folder, "*.bin.png")]
    trainer_params.gen.train.skip_invalid = True
    trainer_params.gen.train.batch_size = 1
    trainer_params.gen.setup.train.num_processes = 1
    
    # Use training data for validation (TrainOnly)
    trainer_params.gen.val.images = trainer_params.gen.train.images
    trainer_params.gen.setup.val.num_processes = 1
    
    # Warmstart parameters
    checkpoint_file = os.path.join(checkpoint_folder, "best.ckpt")
    if not os.path.exists(checkpoint_file):
      checkpoint_file = os.path.join(checkpoint_folder, "best.ckpt.json")
    
    if not os.path.exists(checkpoint_file):
      raise FileNotFoundError(f"Checkpoint not found in: {checkpoint_folder}")
    
    trainer_params.warmstart.model = checkpoint_file
    trainer_params.warmstart.allow_partial = True
    trainer_params.warmstart.trim_graph_name = False
    
    # Codec extension if we have new characters
    if chars:
      charset_file = os.path.join(checkpoint_folder, "extended_charset.txt")
      with open(charset_file, 'w', encoding='utf-8') as f:
        f.write(''.join(chars))
      
      trainer_params.codec.include_files = [charset_file]
      trainer_params.codec.auto_compute = True
      trainer_params.codec.keep_loaded = True
      self.logger.info(f"Extended charset with {len(chars)} characters")
    
    # Disable progress bar for cleaner output
    trainer_params.progress_bar = False
    
    return trainer_params
  
  async def continue_learning_async(self, data_folder: str, checkpoint_folder: str,
                                 network: Optional[str] = None, backup: bool = True,
                                 progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Async continue learning with progress tracking"""
    
    if not LIB_MODE:
      raise RuntimeError("Calamari library not available")
    
    self.logger.info(f"Starting continue learning on: {data_folder}")
    
    # Validate inputs
    if not os.path.exists(data_folder):
      raise FileNotFoundError(f"Data directory not found: {data_folder}")
    
    if not os.path.exists(checkpoint_folder):
      raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_folder}")
    
    # Backup model if requested
    backup_dir = None
    if backup:
      backup_dir = self.backup_model(checkpoint_folder)
    
    try:
      # Collect characters and determine network
      chars = self.collect_chars(data_folder)
      
      # Determine network architecture
      if network is None:
        current_network = self.get_current_network(checkpoint_folder)
        network = current_network  # Keep same network for continue learning
        self.logger.info(f"Using current network: {network}")
      else:
        self.logger.info(f"Using specified network: {network}")
      
      # Create output directory
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      new_model_dir = f"{checkpoint_folder}_extended_{timestamp}"
      os.makedirs(new_model_dir, exist_ok=True)
      
      # Create trainer parameters
      trainer_params = self.create_trainer_params(
        data_folder, checkpoint_folder, network, chars, new_model_dir
      )
      
      if progress_callback:
        await progress_callback("params_created", {"network": network, "chars": len(chars)})
      
      # Run training in thread to avoid blocking
      loop = asyncio.get_event_loop()
      
      def run_training():
        try:
          # Use calamari_train directly with the params
          result = calamari_train(trainer_params)
          return {"success": True, "result": result}
        except Exception as e:
          self.logger.error(f"Training failed: {e}")
          return {"success": False, "error": str(e)}
      
      if progress_callback:
        await progress_callback("training_started", {"model_dir": new_model_dir})
      
      # Run training asynchronously
      result = await loop.run_in_executor(None, run_training)
      
      if result["success"]:
        self.logger.info(f"Training completed successfully: {new_model_dir}")
        
        # Show final codec info if available
        final_params = os.path.join(new_model_dir, "trainer_params.json")
        if os.path.exists(final_params):
          with open(final_params, 'r', encoding='utf-8') as f:
            params = json.load(f)
          if "scenario" in params and "codec" in params["scenario"]:
            codec_info = params["scenario"]["codec"]
            if "charset" in codec_info:
              final_charset = codec_info["charset"]
              self.logger.info(f"Final model charset: {self._format_unicode_chars(final_charset)}")
              self.logger.info(f"Final model classes: {len(final_charset)}")
        
        if progress_callback:
          await progress_callback("training_completed", {"model_dir": new_model_dir})
        
        return {
          "success": True,
          "model_dir": new_model_dir,
          "backup_dir": backup_dir,
          "network": network,
          "chars_count": len(chars)
        }
      else:
        self.logger.error(f"Training failed: {result['error']}")
        if progress_callback:
          await progress_callback("training_failed", {"error": result["error"]})
        
        return {
          "success": False,
          "error": result["error"],
          "backup_dir": backup_dir
        }
    
    except Exception as e:
      self.logger.error(f"Continue learning error: {e}")
      if progress_callback:
        await progress_callback("error", {"error": str(e)})
      
      return {
        "success": False,
        "error": str(e),
        "backup_dir": backup_dir
      }
  
  def continue_learning_sync(self, data_folder: str, checkpoint_folder: str,
                           network: Optional[str] = None, backup: bool = True) -> Dict[str, Any]:
    """Synchronous wrapper for continue learning"""
    try:
      # Try to get current running loop
      loop = asyncio.get_running_loop()
      # If there's already a running loop, we can't use run_until_complete
      # For now, use a simple approach - run the async code in a thread
      import concurrent.futures
      with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, self.continue_learning_async(data_folder, checkpoint_folder, network, backup))
        return future.result()
    except RuntimeError:
      # No running loop, use asyncio.run directly
      return asyncio.run(self.continue_learning_async(data_folder, checkpoint_folder, network, backup))
