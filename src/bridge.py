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

class OCRBridge:
  def __init__(self):
    self.model_dir = "models/new_model"
    self.model_path = os.path.join(self.model_dir, "best.ckpt.json")
    self.learner = CalamariLearner(self.model_path)

  def predict(self, image_path: str) -> Tuple[str, float]:
    return self.learner.do_predict(image_path)

  def train_on_failset(self, failset_dir: str):
    images = [f for f in os.listdir(failset_dir) if f.endswith('.bin.png')]
    if not images: return

    try:
      scenario = CalamariScenario()
      params = scenario.default_trainer_params()
      params.warmstart = self.model_path
      params.output_dir = self.model_dir
      params.epochs = 5
      params.train.images = [os.path.join(failset_dir, "*.bin.png").replace('\\', '/')]
      params.train.gt_extension = ".gt.txt"
      
      print(f"🎓 Training on {len(images)} samples...")
      trainer = scenario.cls().create_trainer(params)
      trainer.train()
      
      # FORCE SAVE
      trainer.scenario.save_model(params.output_dir)
      
      if os.path.exists(self.model_path):
        os.utime(self.model_path, (time.time(), time.time()))
      
      print("✅ Model saved. Reloading...")
      self.learner = CalamariLearner(self.model_path)
    except Exception as e:
      print(f"❌ Training failed: {e}")

  def get_learner_info(self) -> str:
    return "Calamari (Lib Mode) - " + ("READY" if self.learner.predictor else "NO_MODEL")