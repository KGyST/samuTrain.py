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
  def __init__(self, model_path: str, interruption_manager=None):
    self.model_path = model_path
    self.predictor = None
    self.interruption_manager = interruption_manager
    # Extract model directory from model path for continue learning
    self.model_dir = os.path.dirname(model_path)
    # Initialize continue learning engine
    self.learning_engine = ContinueLearningEngine(self.model_dir, interruption_manager)

  def predict(self, image_path: str) -> Tuple[str, float]:
    if not self.predictor:
      self.predictor = MultiPredictor.from_paths(checkpoints=[self.model_path])

    try:
      # Preprocess image
      image = Image.open(image_path).convert('L')
      image_np = np.array(image)

      # predict_raw yields tfaip Sample objects; Sample.outputs is often
      # (list[PredictionResult], voted Prediction) — not a list of results.
      predictions = list(self.predictor.predict_raw([image_np]))

      if not predictions:
        return "ERROR", 0.0

      sample = predictions[0]
      outs = getattr(sample, "outputs", None)
      if outs is not None:
        if isinstance(outs, tuple) and len(outs) > 0:
          pr_list = outs[0]
          if isinstance(pr_list, list) and len(pr_list) > 0:
            bg = pr_list[0]
            if hasattr(bg, "sentence"):
              pobj = getattr(bg, "prediction", None)
              conf = float(getattr(pobj, "avg_char_probability", 0.0) or 0.0) if pobj else 0.0
              return bg.sentence, conf
          if len(outs) >= 2 and outs[1] is not None:
            vote_p = outs[1]
            if hasattr(vote_p, "sentence") and vote_p.sentence:
              return vote_p.sentence, float(getattr(vote_p, "avg_char_probability", 0.0) or 0.0)
        if isinstance(outs, list) and len(outs) > 0:
          best_guess = outs[0]
          if hasattr(best_guess, "sentence"):
            conf = float(getattr(best_guess, "avg_char_probability", 0.0) or 0.0)
            return best_guess.sentence, conf
          if hasattr(best_guess, "prediction"):
            pobj = best_guess.prediction
            return (
              getattr(pobj, "sentence", "") or "",
              float(getattr(pobj, "avg_char_probability", 0.0) or 0.0),
            )

      if hasattr(sample, "sentence") and sample.sentence is not None:
        return sample.sentence, float(getattr(sample, "avg_char_probability", 0.0) or 0.0)

      return "ERROR", 0.0

    except Exception as e:
      print(f"⚠️ OCR Error: {e}")
      return "ERROR", 0.0

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
                       network: Optional[str] = None, backup: bool = True,
                       force: bool = False, epochs: Optional[int] = None) -> Dict[str, Any]:
    """Continue learning using library-based engine"""
    if checkpoint_folder is None:
      checkpoint_folder = self.model_dir
    
    return self.learning_engine.continue_learning_sync(
      data_folder=data_folder,
      checkpoint_folder=checkpoint_folder,
      network=network,
      backup=backup,
      force=force,
      epochs=epochs
    )
  
  async def continue_learning_async(self, data_folder: str, checkpoint_folder: Optional[str] = None,
                                   network: Optional[str] = None, backup: bool = True,
                                   progress_callback: Optional[callable] = None,
                                   force: bool = False, epochs: Optional[int] = None) -> Dict[str, Any]:
    """Async continue learning using library-based engine"""
    if checkpoint_folder is None:
      checkpoint_folder = self.model_dir
    
    return await self.learning_engine.continue_learning_async(
      data_folder=data_folder,
      checkpoint_folder=checkpoint_folder,
      network=network,
      backup=backup,
      progress_callback=progress_callback,
      force=force,
      epochs=epochs
    )
  
  def get_learning_engine(self) -> ContinueLearningEngine:
    """Get the continue learning engine for advanced operations"""
    return self.learning_engine