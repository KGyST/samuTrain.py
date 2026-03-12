import os
import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any
from PIL import Image
from calamari_ocr.ocr.predict.predictor import MultiPredictor
from .learner_interface import LearnerInterface
from .continue_learning import ContinueLearningEngine

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning)

class CalamariLearner(LearnerInterface):
  def __init__(self, model_path: str):
    self.model_path = model_path
    self.predictor = None
    # Extract model directory from model path for continue learning
    self.model_dir = os.path.dirname(model_path)
    # Initialize continue learning engine
    self.learning_engine = ContinueLearningEngine(self.model_dir)

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
        # Debug: print the structure
        pred = predictions[0]
        print(f"Debug: pred type={type(pred)}, outputs type={type(pred.outputs)}")
        if hasattr(pred.outputs, '__len__') and len(pred.outputs) > 0:
          best_guess = pred.outputs[0]
          print(f"Debug: best_guess type={type(best_guess)}")
          return best_guess.sentence, best_guess.avg_char_probability
        else:
          print(f"Debug: outputs content: {pred.outputs}")
          return "ERROR", 0.0
        
    except Exception as e:
      print(f"⚠️ OCR Error: {e}")
      return "ERROR", 0.0
      
    return ("", 0.0)

  def is_available(self) -> bool:
    return os.path.exists(self.model_path)
    
  def get_name(self) -> str:
    return "Calamari OCR"
  
  def do_predict(self, image_path: str) -> Tuple[str, float]:
    """Predict method that matches the old bridge.py API"""
    return self.predict(image_path)
  
  def reload_model(self):
    """Reload the predictor after training"""
    if self.predictor:
      try:
        # Reinitialize the predictor
        self.predictor = MultiPredictor.from_paths(checkpoints=[self.model_path])
        print(f"✅ Model reloaded: {self.model_path}")
      except Exception as e:
        print(f"⚠️ Reload error: {e}")
  
  def continue_learning(self, data_folder: str, checkpoint_folder: Optional[str] = None,
                       network: Optional[str] = None, backup: bool = True) -> Dict[str, Any]:
    """Continue learning using library-based engine"""
    if checkpoint_folder is None:
      checkpoint_folder = self.model_dir
    
    return self.learning_engine.continue_learning_sync(
      data_folder=data_folder,
      checkpoint_folder=checkpoint_folder,
      network=network,
      backup=backup
    )
  
  async def continue_learning_async(self, data_folder: str, checkpoint_folder: Optional[str] = None,
                                   network: Optional[str] = None, backup: bool = True,
                                   progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Async continue learning using library-based engine"""
    if checkpoint_folder is None:
      checkpoint_folder = self.model_dir
    
    return await self.learning_engine.continue_learning_async(
      data_folder=data_folder,
      checkpoint_folder=checkpoint_folder,
      network=network,
      backup=backup,
      progress_callback=progress_callback
    )
  
  def get_learning_engine(self) -> ContinueLearningEngine:
    """Get the continue learning engine for advanced operations"""
    return self.learning_engine