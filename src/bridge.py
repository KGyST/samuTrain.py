import os
import sys
import uuid
import glob
import shutil
import signal
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Protocol
from dataclasses import dataclass, field
from datetime import datetime

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress specific loggers to reduce noise
logging.getLogger('tfaip').setLevel(logging.ERROR)
logging.getLogger('tfaip.device.device_config').setLevel(logging.ERROR)
logging.getLogger('tfaip.trainer.callbacks.benchm').setLevel(logging.INFO)
logging.getLogger('calamari_ocr').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Environment Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Handle Windows Encoding - Fixed approach
if sys.platform == "win32":
    import codecs
    import io
    # Create a more robust wrapper for stdout
    if hasattr(sys.stdout, 'buffer') and not sys.stdout.closed:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    else:
        # Fallback if buffer is not available or closed
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')

# --- Constants ---
UTF_8 = 'utf-8'
BIN_PNG = ".bin.png"
GT_TXT = ".gt.txt"

# --- Configuration ---
@dataclass
class TrainingConfig:
  """Centralized configuration for training parameters."""
  default_network: str = "cnn=8:3x3,pool=2x2,lstm=32"
  default_epochs: int = 5
  validation_split: float = 0.2
  batch_size: int = 1
  num_processes: int = 1
  timeout_graceful: int = 60
  model_dir_env: str = 'SAMUTRAIN_MODEL_FOLDER'
  default_model_dir: str = 'models/new_model'

config = TrainingConfig()

# --- Custom Exceptions ---
class OCRError(Exception):
  """Base exception for OCR operations."""
  pass

class DatasetNotFoundError(OCRError):
  """Raised when dataset files are missing."""
  pass

class DatabaseError(OCRError):
  """Raised when database operations fail."""
  pass

class LearnerNotInitializedError(OCRError):
  """Raised when learner is not properly initialized."""
  pass

# --- Protocol for Database Interface ---
class DatabaseProtocol(Protocol):
  """Protocol defining the database interface required by the bridge."""
  
  def get_training_cases(self, include_gt_only: bool = True) -> List[Dict[str, Any]]:
    """Get training cases from database."""
    ...
  
  def update_predictions_batch(self, predictions: List[Dict[str, Any]]) -> bool:
    """Update predictions in batch."""
    ...
  
  def insert_training_session(self, session_id: str, data_folder: str, 
                              checkpoint_folder: str, network: Optional[str] = None,
                              backup_folder: Optional[str] = None, 
                              chars_count: int = 0) -> int:
    """Insert training session."""
    ...
  
  def update_training_session(self, session_id: str, status: str, 
                              model_folder: Optional[str] = None,
                              error_message: Optional[str] = None) -> None:
    """Update training session."""
    ...

# --- Signal Handler ---
class SignalHandler:
  """Handles system signals for graceful shutdown."""
  
  def __init__(self):
    self.shutdown_event = threading.Event()
    self.training_lock = threading.Lock()
    self.training_process: Optional[subprocess.Popen] = None
  
  def set_training_process(self, process: subprocess.Popen) -> None:
    """Set the current training process."""
    with self.training_lock:
      self.training_process = process
  
  def handle_signal(self, signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    self.shutdown_event.set()
    
    with self.training_lock:
      if self.training_process:
        try:
          if hasattr(signal, 'SIGUSR1'):
            self.training_process.send_signal(signal.SIGUSR1)
            logger.info("Sent save signal (SIGUSR1) to training process.")
          
          logger.info("Terminating training process...")
          self.training_process.terminate()
          try:
            self.training_process.wait(timeout=config.timeout_graceful)
            logger.info("Training terminated gracefully.")
          except subprocess.TimeoutExpired:
            logger.warning("Process did not terminate gracefully. Force killing...")
            self.training_process.kill()
            self.training_process.wait()
        except (AttributeError, OSError) as e:
          logger.error(f"Error terminating process: {e}")
    
    logger.info("Exiting application.")
    sys.exit(0)
  
  def register_handlers(self) -> None:
    """Register signal handlers."""
    signal.signal(signal.SIGINT, self.handle_signal)
    signal.signal(signal.SIGTERM, self.handle_signal)

# Global signal handler instance
signal_handler = SignalHandler()
signal_handler.register_handlers()

# --- Helper Functions ---
def clean_unicode_text(text: str) -> str:
  """Remove invisible Unicode control characters."""
  if not text:
    return text
  
  replacements = {
    '\u202a': '[LTR]', '\u202b': '[RTL]', '\u202c': '[PDF]',
    '\u202d': '[LRO]', '\u202e': '[RLO]'
  }
  
  # Also handle escaped representations if they exist in the string
  for char, repl in replacements.items():
    text = text.replace(char, repl)
    text = text.replace(repr(char)[1:-1], repl)
      
  return text

def normalize_data_path(data_path: str) -> str:
  """Normalize data path to include glob pattern if pointing to a directory."""
  assert isinstance(data_path, str), f"data_path must be string, got {type(data_path)}"
  
  p = Path(data_path)
  
  if "*" in data_path or data_path.endswith(BIN_PNG):
    return data_path
  
  if p.is_dir() or data_path.endswith(os.sep):
    return str(p / f"*{BIN_PNG}")
  
  return data_path

def verify_dataset(data_pattern: str) -> bool:
  """Verify dataset integrity (images and ground truth)."""
  assert isinstance(data_pattern, str), f"data_pattern must be string, got {type(data_pattern)}"
  
  images = glob.glob(data_pattern)
  if not images:
    logger.error(f"No images found matching pattern: {data_pattern}")
    return False
  
  missing_gt = []
  for img in images:
    gt_file = img.replace(BIN_PNG, GT_TXT)
    if not os.path.exists(gt_file):
      missing_gt.append(gt_file)
  
  if missing_gt:
    logger.error(f"Missing Ground Truth files for {len(missing_gt)} images.")
    return False
      
  logger.info(f"Dataset verified: {len(images)} image pairs found.")
  return True

def split_dataset(image_paths: List[str], val_ratio: float) -> Tuple[List[str], List[str]]:
  """Split dataset into training and validation sets."""
  assert isinstance(image_paths, list), "image_paths must be a list"
  assert isinstance(val_ratio, (int, float)), "val_ratio must be numeric"
  assert 0 <= val_ratio <= 1, "val_ratio must be between 0 and 1"
  
  if not image_paths:
    return [], []
  
  # Shuffle to ensure randomness
  shuffled = image_paths.copy()
  import random
  random.shuffle(shuffled)
  
  split_idx = int(len(shuffled) * (1 - val_ratio))
  train_set = shuffled[:split_idx]
  val_set = shuffled[split_idx:]
  
  logger.info(f"Data split: {len(train_set)} train, {len(val_set)} validation")
  return train_set, val_set

def resolve_case_image_file(img_path: str) -> str:
  """Resolve on-disk path for a case image using SAMUTRAIN_DATA_FOLDER."""
  assert isinstance(img_path, str), f"img_path must be string, got {type(img_path)}"
  
  sep_img = img_path.replace("/", os.sep)
  data_folder = os.path.normpath(
    os.path.abspath(os.environ.get("SAMUTRAIN_DATA_FOLDER", "data"))
  )
  parent_data = os.path.dirname(data_folder)

  direct = os.path.join(data_folder, sep_img)
  if os.path.isfile(direct):
    return direct

  if parent_data and os.path.isdir(parent_data):
    under_parent = os.path.normpath(os.path.join(parent_data, sep_img))
    if os.path.isfile(under_parent):
      return under_parent

  base_only = os.path.basename(sep_img)
  if parent_data and os.path.isdir(parent_data) and base_only == sep_img:
    try:
      for name in sorted(os.listdir(parent_data)):
        subdir = os.path.join(parent_data, name)
        if not os.path.isdir(subdir):
          continue
        candidate = os.path.join(subdir, base_only)
        if os.path.isfile(candidate):
          return candidate
    except OSError:
      pass

  return direct

# --- Training Manager ---
class TrainingManager:
  """Manages training operations and parameter setup."""
  
  def __init__(self, db: DatabaseProtocol):
    self.db = db
  
  def setup_training_params(self, output_dir: str, epochs: int, network: str, 
                           train_images: List[str], val_images: List[str]) -> Any:
    """Configure Calamari training parameters."""
    assert os.path.isdir(output_dir), f"Output directory does not exist: {output_dir}"
    assert epochs > 0, f"Epochs must be positive, got {epochs}"
    assert isinstance(network, str) and network, "Network must be non-empty string"
    assert train_images, "Train images list cannot be empty"
    
    try:
      from calamari_ocr.ocr.scenario import CalamariScenario
      trainer_params = CalamariScenario.default_trainer_params()
      
      trainer_params.output_dir = output_dir
      trainer_params.epochs = epochs
      trainer_params.network = network
      
      trainer_params.gen.train.images = train_images
      trainer_params.gen.val.images = val_images
      
      trainer_params.gen.setup.train.batch_size = config.batch_size
      trainer_params.gen.setup.train.num_processes = config.num_processes
      trainer_params.gen.setup.val.num_processes = config.num_processes
      
      trainer_params.codec.auto_compute = True
      trainer_params.progress_bar = False
      
      return trainer_params
    except Exception as e:
      logger.error(f"Failed to setup training parameters: {e}")
      raise OCRError(f"Training parameter setup failed: {e}")
  
  def log_training_session(self, session_id: str, data_folder: str, 
                          checkpoint_folder: str, network: Optional[str] = None) -> None:
    """Log training session to database."""
    try:
      self.db.insert_training_session(
        session_id=session_id,
        data_folder=data_folder,
        checkpoint_folder=checkpoint_folder,
        network=network,
        backup_folder=None,
        chars_count=0
      )
    except Exception as e:
      logger.warning(f"Failed to log session start: {e}")
  
  def update_training_session_status(self, session_id: str, status: str, 
                                   model_folder: Optional[str] = None,
                                   error_message: Optional[str] = None) -> None:
    """Update training session status."""
    try:
      self.db.update_training_session(session_id, status, model_folder, error_message)
    except Exception as e:
      logger.warning(f"Failed to update session status: {e}")

# --- Main Bridge Class ---
class OCRBridge:
  """Main bridge class for OCR operations and training orchestration."""
  
  def __init__(self, db: DatabaseProtocol):
    assert db is not None, "Database instance cannot be None"
    
    self.model_dir = os.environ.get(config.model_dir_env, config.default_model_dir)
    self.model_path = str(Path(self.model_dir) / "best.ckpt.json")
    self.db = db
    self.training_manager = TrainingManager(db)
    
    # Initialize Learner
    self.learner = self._initialize_learner()
  
  def _initialize_learner(self):
    """Initialize the OCR learner."""
    try:
      from engines.calamari_learner import CalamariLearner
      return CalamariLearner(self.model_path)
    except ImportError:
      logger.error("CalamariLearner not found. Ensure dependencies are installed.")
      return None

  def predict(self, image_path: str) -> Tuple[str, float]:
    """Run OCR prediction on an image."""
    assert isinstance(image_path, str), f"image_path must be string, got {type(image_path)}"
    assert os.path.isfile(image_path), f"Image file does not exist: {image_path}"
    
    if not self.learner:
      raise LearnerNotInitializedError("Learner not initialized.")
    return self.learner.do_predict(image_path)

  def start_new_model(self, data_folder: str, model_folder: str, 
                      epochs: Optional[int] = None, network: Optional[str] = None) -> bool:
    """Start training from scratch."""
    assert isinstance(data_folder, str), f"data_folder must be string, got {type(data_folder)}"
    assert isinstance(model_folder, str), f"model_folder must be string, got {type(model_folder)}"
    
    epochs = epochs or config.default_epochs
    network = network or config.default_network
    
    logger.info(f"🚀 Starting initial training: {epochs} epochs, network: {network}")
    
    # Validate inputs
    if not os.path.exists(data_folder):
      logger.error(f"Data folder not found: {data_folder}")
      return False
      
    # Get data from DB
    try:
      training_cases = self.db.get_training_cases(include_gt_only=True)
    except Exception as e:
      logger.error(f"Database error: {e}")
      return False

    if not training_cases:
      logger.error("No training cases found in database.")
      return False

    # Prepare paths
    model_folder = str(Path(model_folder).resolve())
    os.makedirs(model_folder, exist_ok=True)
    
    image_paths = [case['full_img_path'] for case in training_cases]
    
    # Split data
    train_imgs, val_imgs = split_dataset(image_paths, config.validation_split)
    
    if not train_imgs:
      logger.error("Not enough data for training after split.")
      return False

    try:
      trainer_params = self.training_manager.setup_training_params(
        model_folder, epochs, network, train_imgs, val_imgs
      )
      
      # Log session to DB
      session_id = str(uuid.uuid4())
      self.training_manager.log_training_session(session_id, data_folder, model_folder, network)
      
      try:
        from calamari_ocr.ocr.training.trainer import main as calamari_train
      except ImportError as e:
        logger.error(f"Failed to import calamari training module: {e}")
        raise OCRError("Calamari training module not available")
      result = calamari_train(trainer_params)
      
      logger.info("✅ Initial training completed.")
      
      # Update predictions
      self._update_predictions_batch(training_cases)
      self.learner.reload_model()
      
      # Update session status
      self.training_manager.update_training_session_status(session_id, 'completed', model_folder)
      return True
      
    except Exception as e:
      logger.error(f"❌ Training failed: {e}", exc_info=True)
      
      # Update session status
      session_id = str(uuid.uuid4())
      self.training_manager.update_training_session_status(session_id, 'failed', error_message=str(e))
      return False

  def continue_learning(self, data_folder: str, checkpoint_folder: str, 
                        network: Optional[str] = None, backup: bool = True, 
                        force: bool = False, epochs: Optional[int] = None) -> bool:
    """Continue learning from an existing model."""
    assert isinstance(data_folder, str), f"data_folder must be string, got {type(data_folder)}"
    assert isinstance(checkpoint_folder, str), f"checkpoint_folder must be string, got {type(checkpoint_folder)}"
    
    session_id = str(uuid.uuid4())
    logger.info(f"🎓 Continue learning session: {session_id[:8]}")
    
    # Log session to DB
    self.training_manager.log_training_session(session_id, data_folder, checkpoint_folder, network)

    try:
      if not self.learner:
        raise LearnerNotInitializedError("Learner not initialized.")
          
      result = self.learner.continue_learning(
        data_folder=data_folder,
        checkpoint_folder=checkpoint_folder,
        network=network,
        backup=backup,
        force=force,
        epochs=epochs
      )

      if result.get("success"):
        logger.info("✅ Continue learning successful.")
        self.training_manager.update_training_session_status(
          session_id, 'completed', result.get('model_dir')
        )
        
        # Update predictions in database after training
        try:
          training_cases = self.db.get_training_cases(include_gt_only=True)
          self._update_predictions_batch(training_cases)
        except Exception as e:
          logger.warning(f"Failed to update predictions: {e}")
        
        self.learner.reload_model()
        return True
      else:
        logger.error(f"❌ Continue learning failed: {result.get('error')}")
        self.training_manager.update_training_session_status(
          session_id, 'failed', error_message=result.get('error')
        )
        return False
          
    except Exception as e:
      logger.error(f"❌ Error in continue learning: {e}", exc_info=True)
      self.training_manager.update_training_session_status(
        session_id, 'failed', error_message=str(e)
      )
      return False

  def train_on_failset(self, failset_dir: str) -> bool:
    """Train specifically on failed cases."""
    assert isinstance(failset_dir, str), f"failset_dir must be string, got {type(failset_dir)}"
    
    if not self.learner:
      return False
      
    logger.info("🎓 Training on failset cases...")
    
    try:
      failset_cases = [c for c in self.db.get_training_cases(include_gt_only=True) if c.get('is_failset')]
      if not failset_cases:
        logger.warning("No failset cases found.")
        return False
        
      logger.info(f"Found {len(failset_cases)} failset cases.")
      
      # Prepare paths
      failset_paths = [c['full_img_path'] for c in failset_cases]
      train_imgs, val_imgs = split_dataset(failset_paths, config.validation_split)
      
      trainer_params = self.training_manager.setup_training_params(
        self.model_dir, 10, config.default_network, train_imgs, val_imgs
      )
      
      try:
        from calamari_ocr.ocr.training.trainer import main as calamari_train
      except ImportError as e:
        logger.error(f"Failed to import calamari training module: {e}")
        raise OCRError("Calamari training module not available")
      calamari_train(trainer_params)
      
      logger.info("✅ Failset training completed.")
      self._update_predictions_batch(failset_cases)
      self.learner.reload_model()
      return True
      
    except Exception as e:
      logger.error(f"❌ Failset training failed: {e}", exc_info=True)
      return False

  def _update_predictions_batch(self, cases: List[Dict]) -> None:
    """Update predictions for a list of cases."""
    assert isinstance(cases, list), "cases must be a list"
    
    predictions = []
    for case in cases:
      try:
        pred, conf = self.predict(case['full_img_path'])
        predictions.append({
          'case_id': case['id'],
          'ocr_text': clean_unicode_text(pred),
          'confidence': conf
        })
      except Exception as e:
        logger.warning(f"Failed to predict case {case['id']}: {e}")
    
    if predictions:
      try:
        self.db.update_predictions_batch(predictions)
        logger.info(f"Updated {len(predictions)} predictions in database.")
      except Exception as e:
        logger.error(f"Failed to update database: {e}")

  def get_learner_status(self) -> str:
    """Get the current status of the learner."""
    if not self.learner:
      return "NOT INITIALIZED"
    return f"Calamari - {'READY' if self.learner.predictor else 'NO MODEL'}"

# --- Legacy Compatibility Functions ---
def insert_training_session(session_id: str, data_folder: str, checkpoint_folder: str,
                          network: Optional[str] = None, backup_folder: Optional[str] = None,
                          chars_count: int = 0) -> int:
  """Legacy function for inserting training session."""
  from db import db
  return db.insert_training_session(session_id, data_folder, checkpoint_folder, 
                                   network, backup_folder, chars_count)

def update_training_session(session_id: str, status: str, model_folder: Optional[str] = None,
                          error_message: Optional[str] = None) -> None:
  """Legacy function for updating training session."""
  from db import db
  db.update_training_session(session_id, status, model_folder, error_message)

# Entry point example
if __name__ == "__main__":
  # Example usage
  from db import db
  bridge = OCRBridge(db)
  # Uncomment to test
  # bridge.start_new_model("./data", "./models/test_model")

    