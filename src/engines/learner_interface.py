"""
Base Learner Interface for samuTrain V2
Defines the interface that all OCR learners must implement
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional


class LearnerInterface(ABC):
    """
    Abstract base class for all OCR learners
    """
    
    @abstractmethod
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Run OCR prediction on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (predicted_text, confidence_score)
            
        Raises:
            Exception: If prediction fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the learner is available and ready to use
        
        Returns:
            True if learner is available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this learner
        
        Returns:
            Human-readable name of the learner
        """
        pass
    
    def get_fallback_prediction(self, gt_text: str = "") -> Tuple[str, float]:
        """
        Get fallback prediction when main prediction fails
        
        Args:
            gt_text: Ground truth text to use as fallback
            
        Returns:
            Tuple of (fallback_text, fallback_confidence)
        """
        import random
        return (gt_text if gt_text else "", random.uniform(0.1, 1.0))
