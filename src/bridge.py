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
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Environment Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Handle Windows Encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

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

# --- Mock Database Interface (Replace with actual import) ---
# In production, replace this with: from db import db, insert_training_session, update_training_session
class MockDB:
    def get_training_cases(self, include_gt_only: bool = False) -> List[Dict[str, Any]]:
        # Placeholder: Return empty list or mock data
        return []
    
    def update_predictions_batch(self, predictions: List[Dict]) -> None:
        pass

# Simulating the real DB imports
try:
    from db import insert_training_session, update_training_session, db
except ImportError:
    logger.warning("Database module not found. Using mock DB for demonstration.")
    db = MockDB()
    # Define dummy functions if import fails
    def insert_training_session(**kwargs): pass
    def update_training_session(**kwargs): pass

# --- Signal Handling ---
shutdown_event = threading.Event()
training_lock = threading.Lock()
training_process: Optional[subprocess.Popen] = None

def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    global training_process
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_event.set()
    
    with training_lock:
        if training_process:
            try:
                if hasattr(signal, 'SIGUSR1'):
                    training_process.send_signal(signal.SIGUSR1)
                    logger.info("Sent save signal (SIGUSR1) to training process.")
                
                logger.info("Terminating training process...")
                training_process.terminate()
                try:
                    training_process.wait(timeout=config.timeout_graceful)
                    logger.info("Training terminated gracefully.")
                except subprocess.TimeoutExpired:
                    logger.warning("Process did not terminate gracefully. Force killing...")
                    training_process.kill()
                    training_process.wait()
            except (AttributeError, OSError) as e:
                logger.error(f"Error terminating process: {e}")
    
    logger.info("Exiting application.")
    sys.exit(0)

# Register handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
    p = Path(data_path)
    
    if "*" in data_path or data_path.endswith(".bin.png"):
        return data_path
    
    if p.is_dir() or data_path.endswith(os.sep):
        return str(p / "*.bin.png")
    
    return data_path

def verify_dataset(data_pattern: str) -> bool:
    """Verify dataset integrity (images and ground truth)."""
    images = glob.glob(data_pattern)
    if not images:
        logger.error(f"No images found matching pattern: {data_pattern}")
        return False
    
    missing_gt = []
    for img in images:
        gt_file = img.replace(".bin.png", ".gt.txt")
        if not os.path.exists(gt_file):
            missing_gt.append(gt_file)
    
    if missing_gt:
        logger.error(f"Missing Ground Truth files for {len(missing_gt)} images.")
        return False
        
    logger.info(f"Dataset verified: {len(images)} image pairs found.")
    return True

def split_dataset(image_paths: List[str], val_ratio: float) -> Tuple[List[str], List[str]]:
    """Split dataset into training and validation sets."""
    if not image_paths:
        return [], []
    
    # Shuffle to ensure randomness
    shuffled = image_paths.copy()
    # Simple shuffle implementation (or use random.shuffle)
    import random
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * (1 - val_ratio))
    train_set = shuffled[:split_idx]
    val_set = shuffled[split_idx:]
    
    logger.info(f"Data split: {len(train_set)} train, {len(val_set)} validation")
    return train_set, val_set

# --- Main Class ---
class OCRBridge:
  def __init__(self):
        self.model_dir = os.environ.get(config.model_dir_env, config.default_model_dir)
        self.model_path = str(Path(self.model_dir) / "best.ckpt.json")
        
        # Initialize Learner (Assuming CalamariLearner is available)
        try:
            from engines.calamari_learner import CalamariLearner
            self.learner = CalamariLearner(self.model_path)
        except ImportError:
            logger.error("CalamariLearner not found. Ensure dependencies are installed.")
            self.learner = None

  def predict(self, image_path: str) -> Tuple[str, float]:
        if not self.learner:
            raise OCRError("Learner not initialized.")
        return self.learner.do_predict(image_path)

  def _setup_training_params(self, output_dir: str, epochs: int, network: str, 
                               train_images: List[str], val_images: List[str]) -> Any:
        """Configure Calamari training parameters."""
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
            raise

  def start_new_model(self, data_folder: str, model_folder: str, 
                        epochs: Optional[int] = None, network: Optional[str] = None) -> bool:
        """Start training from scratch."""
        epochs = epochs or config.default_epochs
        network = network or config.default_network
        
        logger.info(f"🚀 Starting initial training: {epochs} epochs, network: {network}")
        
        # Validate inputs
        if not os.path.exists(data_folder):
            logger.error(f"Data folder not found: {data_folder}")
            return False
            
        # Get data from DB
        try:
            training_cases = db.get_training_cases(include_gt_only=True)
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
            trainer_params = self._setup_training_params(
                model_folder, epochs, network, train_imgs, val_imgs
            )
            
            from calamari_ocr.ocr.training.trainer import main as calamari_train
            result = calamari_train(trainer_params)
            
            logger.info("✅ Initial training completed.")
            
            # Update predictions
            self._update_predictions_batch(training_cases)
            self.learner.reload_model()
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            return False

  def continue_learning(self, data_folder: str, checkpoint_folder: str, 
                          network: Optional[str] = None, backup: bool = True, 
                          force: bool = False) -> bool:
        """Continue learning from an existing model."""
        session_id = str(uuid.uuid4())
        logger.info(f"🎓 Continue learning session: {session_id[:8]}")
        
        # Log session to DB
        try:
            insert_training_session(
                session_id=session_id,
                data_folder=data_folder,
                checkpoint_folder=checkpoint_folder,
                network=network,
                backup_folder=None,
                chars_count=0
            )
        except Exception as e:
            logger.warning(f"Failed to log session start: {e}")

        try:
            if not self.learner:
                raise OCRError("Learner not initialized.")
                
            result = self.learner.continue_learning(
                data_folder=data_folder,
                checkpoint_folder=checkpoint_folder,
                network=network,
                backup=backup,
                force=force
            )
            
            if result.get("success"):
                logger.info("✅ Continue learning successful.")
                update_training_session(session_id, status='completed', model_folder=result.get('model_dir'))
                
                # Update predictions in database after training
                try:
                    training_cases = db.get_training_cases(include_gt_only=True)
                    self._update_predictions_batch(training_cases)
                except Exception as e:
                    logger.warning(f"Failed to update predictions: {e}")
                
                self.learner.reload_model()
                return True
            else:
                logger.error(f"❌ Continue learning failed: {result.get('error')}")
                update_training_session(session_id, status='failed', error_message=result.get('error'))
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in continue learning: {e}", exc_info=True)
            update_training_session(session_id, status='failed', error_message=str(e))
            return False

  def train_on_failset(self, failset_dir: str) -> bool:
        """Train specifically on failed cases."""
        if not self.learner:
            return False
            
        logger.info("🎓 Training on failset cases...")
        
        try:
            failset_cases = [c for c in db.get_training_cases(include_gt_only=True) if c.get('is_failset')]
            if not failset_cases:
                logger.warning("No failset cases found.")
                return False
                
            logger.info(f"Found {len(failset_cases)} failset cases.")
            
            # Prepare paths
            failset_paths = [c['full_img_path'] for c in failset_cases]
            train_imgs, val_imgs = split_dataset(failset_paths, config.validation_split)
            
            trainer_params = self._setup_training_params(
                self.model_dir, 10, config.default_network, train_imgs, val_imgs
            )
            
            from calamari_ocr.ocr.training.trainer import main as calamari_train
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
                db.update_predictions_batch(predictions)
                logger.info(f"Updated {len(predictions)} predictions in database.")
            except Exception as e:
                logger.error(f"Failed to update database: {e}")

  def get_learner_status(self) -> str:
        if not self.learner:
            return "NOT INITIALIZED"
        return f"Calamari - {'READY' if self.learner.predictor else 'NO MODEL'}"

# Entry point example
if __name__ == "__main__":
    # Example usage
    bridge = OCRBridge()
    # Uncomment to test
    # bridge.start_new_model("./data", "./models/test_model")

    