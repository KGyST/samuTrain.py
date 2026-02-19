import os
import shutil
import time
import numpy as np
from PIL import Image
from typing import Tuple
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

class CalamariLearner:
  def __init__(self, model_path: str):
    self.model_path = model_path
    self.predictor = None
    if LIB_MODE:
      try:
        params = PredictorParams(silent=True)
        checkpoint_base = self.model_path.replace('.json', '')
        self.predictor = Predictor.from_checkpoint(params, checkpoint=checkpoint_base)
        print(f"✅ Calamari loaded: {checkpoint_base}")
      except Exception as e:
        print(f"⚠️ Load error: {e}")

  def do_predict(self, image_path: str) -> Tuple[str, float]:
    if not self.predictor: return "ERROR", 0.0
    try:
      img = np.array(Image.open(image_path).convert('L'))
      for sample in self.predictor.predict_raw([img]):
        return sample.sentence, sample.avg_char_probability
    except Exception as e:
      print(f"❌ Prediction failed: {e}")
      return "FAIL", 0.0

  def reload_model(self):
    """Reload the predictor after training"""
    if LIB_MODE:
      try:
        params = PredictorParams(silent=True)
        checkpoint_base = self.model_path.replace('.json', '')
        self.predictor = Predictor.from_checkpoint(params, checkpoint=checkpoint_base)
        print(f"✅ Model reloaded: {checkpoint_base}")
      except Exception as e:
        print(f"⚠️ Reload error: {e}")

class OCRBridge:
  def __init__(self):
    self.model_dir = "models/new_model"
    self.model_path = os.path.join(self.model_dir, "best.ckpt.json")
    self.learner = CalamariLearner(self.model_path)

  def predict(self, image_path: str) -> Tuple[str, float]:
    return self.learner.do_predict(image_path)

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
      trainer_params.epochs = 5
      trainer_params.gen.train.images = [os.path.join(failset_dir, "*.bin.png")]
      trainer_params.gen.val.images = []  # No validation set for failset training
      trainer_params.gen.setup.train.batch_size = 1
      trainer_params.gen.setup.train.num_processes = 1
      trainer_params.gen.setup.val.num_processes = 1
      
      # Performance optimizations for Windows
      trainer_params.progress_bar = True
      
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