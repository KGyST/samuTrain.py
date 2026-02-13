"""
Mock Calamari OCR Learner Implementation
Simulates Calamari OCR behavior for testing purposes
"""

import os
import random
from typing import Tuple

from .learner_interface import LearnerInterface


class MockCalamariLearner(LearnerInterface):
    """
    Mock Calamari OCR implementation for testing
    Simulates OCR predictions with realistic confidence scores
    """
    
    def __init__(self, model_path: str = ""):
        """
        Initialize Mock Calamari learner
        
        Args:
            model_path: Not used in mock implementation
        """
        self.model_path = model_path
        # Simulate different prediction patterns
        self.prediction_patterns = [
            ("", 0.1),  # Empty prediction with low confidence
            ("", 0.8),  # Empty prediction with high confidence  
            ("244.", 0.9),  # Sample prediction with high confidence
            ("123", 0.7),   # Sample prediction with medium confidence
            ("abc", 0.6),   # Sample prediction with medium confidence
        ]
        
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Simulate OCR prediction on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (predicted_text, confidence_score)
        """
        # Simulate realistic OCR behavior
        base_name = os.path.basename(image_path).replace('.bin.png', '').replace('.png', '')
        
        # Use hash of filename for consistent predictions
        seed = hash(base_name) % 1000
        random.seed(seed)
        
        # 70% chance of empty prediction (simulating untrained model)
        if random.random() < 0.7:
            confidence = random.uniform(0.1, 0.9)
            return ("", confidence)
        
        # 30% chance of some text prediction
        sample_texts = ["244.", "123", "abc", "test", "001"]
        predicted_text = random.choice(sample_texts)
        confidence = random.uniform(0.6, 0.95)
        
        return (predicted_text, confidence)
    
    def is_available(self) -> bool:
        """Check if Mock Calamari is available (always true)"""
        return True
    
    def get_name(self) -> str:
        """Get the learner name"""
        return "Mock Calamari OCR"
