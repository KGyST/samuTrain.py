import os
from typing import Tuple
from calamari_ocr.ocr.predict.predictor import MultiPredictor
from .learner_interface import LearnerInterface

class CalamariLearner(LearnerInterface):
  def __init__(self, model_path: str):
    self.model_path = model_path
    self.predictor = None

  def predict(self, image_path: str) -> Tuple[str, float]:
    if not self.predictor:
      # Simple loading from pip-installed package
      self.predictor = MultiPredictor.from_paths(checkpoints=[self.model_path])
    
    # Python 3.13 standard generator iteration
    predictions = self.predictor.predict([image_path])
    for pred in predictions:
      return pred.sentence, pred.avg_char_probability
    return ("", 0.0)

  def is_available(self) -> bool:
    return os.path.exists(self.model_path + ".json")

  def get_name(self) -> str:
    return "Calamari OCR (3.13 Clean)"