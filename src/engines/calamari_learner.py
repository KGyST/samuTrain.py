"""
Calamari OCR Learner Implementation
Implements the LearnerInterface for Calamari OCR
"""

import os
import sys
from typing import Tuple, Optional, Dict, Any, List

# Add lib/calamari to path for local calamari
calamari_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lib', 'calamari'))
if calamari_path not in sys.path:
    sys.path.insert(0, calamari_path)

# Import calamari modules (using local version with patches)
import calamari_ocr
from calamari_ocr.ocr.predict.predictor import MultiPredictor
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
            '..', '..', 'lib', 'calamari', 'calamari_ocr', 'test', 'models', 'version3', '0.ckpt'
        )
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Run Calamari OCR prediction on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (predicted_text, confidence_score)
        """
        # Initialize predictor if not already done
        if self.predictor is None:
            # Calamari 2.3.x uses MultiPredictor for checkpoint loading
            # The checkpoints parameter MUST be a list [path]
            self.predictor = MultiPredictor.from_paths(
                checkpoints=[self.model_path]
            )
        
        # Run prediction
        predictions = self.predictor.predict([image_path])
        
        for pred in predictions[0]:
            ocr_text = pred.sentence
            confidence = pred.avg_char_probability
            return (ocr_text, confidence)
        
        return ("", 0.0)
    
    def is_available(self) -> bool:
        """Check if Calamari is available"""
        return os.path.exists(self.model_path)
    
    def get_name(self) -> str:
        """Get the learner name"""
        return "Calamari OCR"
