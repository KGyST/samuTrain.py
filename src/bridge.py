import os
import subprocess
import shutil
import time
import numpy as np
from PIL import Image
from typing import Tuple
import sys
import uuid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'

try:
  from calamari_ocr.ocr.predict.predictor import Predictor
  from calamari_ocr.ocr.predict.params import PredictorParams
  from calamari_ocr.ocr.training.trainer import Trainer
  from calamari_ocr.ocr.training.params import TrainerParams
  from calamari_ocr.ocr.scenario import CalamariScenario
  from calamari_ocr.scripts.train import main as calamari_train
  LIB_MODE = True
except ImportError:
  LIB_MODE = False

# Import database functions for training tracking
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import insert_training_session, update_training_session
from engines.calamari_learner import CalamariLearner

class OCRBridge:
  def __init__(self):
    # Use model folder from environment or default to new_model
    self.model_dir = os.environ.get('SAMUTRAIN_MODEL_FOLDER', 'models/new_model')
    self.model_path = os.path.join(self.model_dir, "best.ckpt.json")
    self.learner = CalamariLearner(self.model_path)

  def predict(self, image_path: str) -> Tuple[str, float]:
    return self.learner.do_predict(image_path)

  def continue_learning(self, data_folder, checkpoint_folder, network=None, backup=True):
    """Use the library-based continue learning engine with database tracking"""
    session_id = str(uuid.uuid4())
    print(f"🎓 Continue learning started on: {data_folder} (session: {session_id[:8]})")
    
    # Insert training session into database
    try:
      insert_training_session(
        session_id=session_id,
        data_folder=data_folder,
        checkpoint_folder=checkpoint_folder,
        network=network,
        backup_folder=None,  # Will be updated after backup
        chars_count=0  # Will be updated after character collection
      )
    except Exception as e:
      print(f"⚠️ Failed to create training session: {e}")
    
    try:
      result = self.learner.continue_learning(
        data_folder=data_folder,
        checkpoint_folder=checkpoint_folder,
        network=network,
        backup=backup
      )
      
      if result["success"]:
        print("✅ Continue learning completed successfully")
        print(f"New model location: {result.get('model_dir')}")
        if result.get('backup_dir'):
          print(f"Model backup: {result['backup_dir']}")
        print(f"Network: {result.get('network')}")
        print(f"Characters learned: {result.get('chars_count', 0)}")
        
        # Update database with success
        try:
          update_training_session(
            session_id=session_id,
            status='completed',
            model_folder=result.get('model_dir')
          )
        except Exception as e:
          print(f"⚠️ Failed to update training session: {e}")
        
        # Reload model after training
        self.learner.reload_model()
        return True
      else:
        print(f"❌ Continue learning failed: {result.get('error')}")
        
        # Update database with failure
        try:
          update_training_session(
            session_id=session_id,
            status='failed',
            error_message=result.get('error')
          )
        except Exception as e:
          print(f"⚠️ Failed to update training session: {e}")
        
        return False
        
    except Exception as e:
      print(f"❌ Continue learning error: {e}")
      
      # Update database with error
      try:
        update_training_session(
          session_id=session_id,
          status='failed',
          error_message=str(e)
        )
      except Exception as db_e:
        print(f"⚠️ Failed to update training session: {db_e}")
      
      return False

  def train_on_failset(self, failset_dir):
    """Uses Calamari library to train/upgrade the model safely"""
    if not LIB_MODE:
      print("❌ Calamari library not available")
      return False
      
    print(f"🎓 Training started on: {failset_dir}")
    
    try:
      # Create default trainer params
      trainer_params = CalamariScenario.default_trainer_params()
      
      # Set the training parameters equivalent to the command line
      trainer_params.output_dir = self.model_dir
      trainer_params.epochs = 10  # Increase epochs for more learning
      trainer_params.gen.train.images = [os.path.join(failset_dir, "*.bin.png")]
      trainer_params.gen.val.images = [os.path.join("data/single_case", "*.bin.png")]  # Use single_case as validation
      trainer_params.gen.setup.train.batch_size = 1
      trainer_params.gen.setup.train.num_processes = 1  # Single process for stability
      trainer_params.gen.setup.val.num_processes = 1
      
      # Performance optimizations
      trainer_params.progress_bar = False  # Disable progress bar to reduce output
      
      # Run the training
      result = calamari_train(trainer_params)
      print("✅ Training completed, model updated.")
      
      # Reload model after training
      self.learner.reload_model()
      return True
    except Exception as e:
      print(f"❌ Library Training failed: {e}")
      return False

  def get_learner_info(self) -> str:
    return "Calamari (Lib Mode) - " + ("READY" if self.learner.predictor else "NO_MODEL")