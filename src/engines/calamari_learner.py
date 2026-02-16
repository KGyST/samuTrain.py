import os
import numpy as np
import warnings
from typing import Tuple
from PIL import Image
from calamari_ocr.ocr.predict.predictor import MultiPredictor
from .learner_interface import LearnerInterface

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning)

class CalamariLearner(LearnerInterface):
  def __init__(self, model_path: str):
    self.model_path = model_path
    self.predictor = None

  def predict(self, image_path: str) -> Tuple[str, float]:
    if not self.predictor:
      self.predictor = MultiPredictor.from_paths(checkpoints=[self.model_path])

    try:
      # Preprocess image
      image = Image.open(image_path).convert('L')
      image_np = np.array(image)

      # Calamari 2.2 API: returns a list of Prediction objects
      # Each Prediction has an 'outputs' list containing PredictionResult objects
      predictions = list(self.predictor.predict_raw([image_np]))
      
      if predictions and len(predictions) > 0:
        # best_guess is the first PredictionResult in the first Prediction
        best_guess = predictions[0].outputs[0]
        return best_guess.sentence, best_guess.avg_char_probability
        
    except Exception as e:
      print(f"⚠️ OCR Error: {e}")
      return "ERROR", 0.0
      
    return ("", 0.0)

  def is_available(self) -> bool:
    return os.path.exists(self.model_path)
    
  def get_name(self) -> str:
    return "Calamari OCR"