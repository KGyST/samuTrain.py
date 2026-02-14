"""
Calamari OCR Learner Implementation
Implements the LearnerInterface for Calamari OCR
"""

import os
import sys
os.environ['TF_USE_LEGACY_KERAS'] = '1'
from typing import Tuple, Optional, Dict, Any, List

# TensorFlow/Keras Bridge: Use legacy Keras 2 setup
import tensorflow as tf
import keras
import tf_keras
sys.modules['tensorflow.keras'] = keras
sys.modules['tf_keras'] = tf_keras

# Add lib/calamari to path for local calamari
calamari_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lib', 'calamari'))
if calamari_path not in sys.path:
    sys.path.insert(0, calamari_path)

# Add lib to path for samuTeszt
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# Import calamari modules (using local version with patches)
import calamari_ocr
from calamari_ocr.ocr.predict.predictor import MultiPredictor
from samuTeszt_python_0_03 import Dumper
CALAMARI_AVAILABLE = True
print("✅ Using local calamari-ocr with TensorFlow compatibility patches")

from .learner_interface import LearnerInterface


class CalamariLearner(LearnerInterface):
    """
    Calamari OCR implementation of LearnerInterface
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Calamari learner
        
        Args:
            model_path: Path to pre-trained model file
        """
        self.model_path = model_path or self._get_default_model_path()
        self.predictor = None
        
    def _get_default_model_path(self) -> str:
        """Get default pre-trained model path"""
        return os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'lib', 'calamari', 'calamari_ocr', 'test', 'models', 'best.ckpt'
        )
    

    def predict(self, image_path: str) -> Tuple[str, float]:
        if self.predictor is None:
          self.predictor = MultiPredictor.from_paths(
            checkpoints=[self.model_path], auto_update_checkpoints=True
          )
        
        # Fix: Inject correct data params object to avoid 'list' object error
        # The dump showed generator_params was accidentally stored as a list
        from calamari_ocr.ocr.dataset.params import DataParams
        for model in self.predictor.models:
          if isinstance(model.data_params.generator_params, list):
            # Reset to default params object which contains the .create() method
            model.data_params = DataParams()

        predictions = self.predictor.predict([image_path])
        
        for pred in predictions:
          return pred.sentence, pred.avg_char_probability
                  
        return ("", 0.0)
    
    def is_available(self) -> bool:
        """Check if Calamari is available"""
        return os.path.exists(self.model_path)
    
    def get_name(self) -> str:
        """Get the learner name"""
        return "Calamari OCR"
