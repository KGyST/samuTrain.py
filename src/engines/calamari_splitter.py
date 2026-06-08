"""
Calamari Character Splitter
---------------------------
Native implementation for splitting word images into character images using Calamari's prediction API.
Eliminates subprocess dependency and provides robust error handling with fallback mechanisms.
"""

import os
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import cv2

# Calamari imports
from calamari_ocr.ocr.predict.predictor import MultiPredictor
from calamari_ocr.ocr.predict.params import Predictions, Prediction, PredictionPosition, PredictionCharacter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SplitterConfig:
    """Configuration for the character splitter"""
    use_native_api: bool = True
    fallback_to_simple: bool = True
    min_char_width: int = 5
    max_char_width: int = 100
    confidence_threshold: float = 0.5
    batch_size: int = 8
    cache_predictions: bool = True

class CalamariCharacterSplitter:
    """Native Calamari character splitter using prediction API"""
    
    def __init__(self, model_path: str, config: SplitterConfig = None):
        """
        Initialize the splitter with a Calamari model
        
        Args:
            model_path: Path to Calamari checkpoint (.json file)
            config: Splitter configuration options
        """
        self.model_path = model_path
        self.config = config or SplitterConfig()
        self.predictor = None
        self._prediction_cache = {} if self.config.cache_predictions else None
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Initialize predictor
        self._initialize_predictor()
    
    def _initialize_predictor(self):
        """Initialize the Calamari predictor"""
        try:
            self.predictor = MultiPredictor.from_paths(checkpoints=[self.model_path])
            logger.info(f"Calamari predictor initialized with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            if not self.config.fallback_to_simple:
                raise
            self.predictor = None
    
    def split_word_to_chars(self, image: np.ndarray, target_text: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Split a word image into individual character images
        
        Args:
            image: Input word image as numpy array (grayscale)
            target_text: Optional target text for fallback simple splitting
            
        Returns:
            List of tuples: (character_image, character_text, confidence)
        """
        if self.config.use_native_api and self.predictor is not None:
            try:
                return self._split_with_calamari_api(image)
            except Exception as e:
                logger.warning(f"Calamari API splitting failed: {e}")
                if not self.config.fallback_to_simple:
                    raise
        
        # Fallback to simple splitting
        if target_text:
            return self._split_with_simple_method(image, target_text)
        else:
            raise ValueError("No target text provided for fallback simple splitting")
    
    def split_batch(self, images: List[np.ndarray], target_texts: List[str] = None) -> List[List[Tuple[np.ndarray, str, float]]]:
        """
        Split multiple word images in batch
        
        Args:
            images: List of word images
            target_texts: Optional list of target texts for fallback
            
        Returns:
            List of character split results for each image
        """
        if not self.config.use_native_api or self.predictor is None:
            # Process individually for simple splitting
            results = []
            for i, image in enumerate(images):
                target_text = target_texts[i] if target_texts else None
                results.append(self.split_word_to_chars(image, target_text))
            return results
        
        try:
            return self._split_batch_with_calamari_api(images)
        except Exception as e:
            logger.warning(f"Batch splitting failed: {e}")
            # Fallback to individual processing
            results = []
            for i, image in enumerate(images):
                target_text = target_texts[i] if target_texts else None
                results.append(self.split_word_to_chars(image, target_text))
            return results
    
    def _split_with_calamari_api(self, image: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        """Split using Calamari's native prediction API"""
        # Check cache first
        if self._prediction_cache is not None:
            img_hash = hash(image.tobytes())
            if img_hash in self._prediction_cache:
                return self._prediction_cache[img_hash]
        
        # Ensure image is in correct format
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get prediction (extended data is not available via predict_raw API)
        predictions = list(self.predictor.predict_raw([image]))
        
        if not predictions:
            raise ValueError("No predictions returned from Calamari")
        
        sample = predictions[0]
        outputs = getattr(sample, "outputs", None)
        
        if outputs is None:
            raise ValueError("No outputs in prediction result")
        
        # Extract prediction results
        prediction_results = []
        if isinstance(outputs, tuple) and len(outputs) > 0:
            prediction_results = outputs[0]
        elif isinstance(outputs, list):
            prediction_results = outputs
        else:
            raise ValueError("Unexpected output format")
        
        if not prediction_results:
            raise ValueError("Empty prediction results")
        
        # Get the first prediction (or voted result if available)
        prediction = prediction_results[0] if isinstance(prediction_results, list) else prediction_results
        
        if not hasattr(prediction, 'prediction') or not prediction.prediction:
            raise ValueError("No prediction data available")
        
        pred_obj = prediction.prediction
        
        # Since native API doesn't provide position data, use hybrid approach
        char_images = []
        
        # Try to extract character predictions and use simple splitting with better confidence
        if hasattr(pred_obj, 'sentence') and pred_obj.sentence:
            predicted_text = pred_obj.sentence
            confidence = getattr(pred_obj, 'avg_char_probability', 0.5)
            
            # Use simple splitting but with predicted text
            char_images = self._split_with_simple_method(image, predicted_text)
            
            # Update confidence for each character
            char_images = [(char_img, char_text, confidence) for char_img, char_text in char_images]
        else:
            raise ValueError("No prediction text available")
        
        if not char_images:
            raise ValueError("No valid character splits found")
        
        # Cache result if enabled
        if self._prediction_cache is not None:
            img_hash = hash(image.tobytes())
            self._prediction_cache[img_hash] = char_images
        
        return char_images
    
    def _split_batch_with_calamari_api(self, images: List[np.ndarray]) -> List[List[Tuple[np.ndarray, str, float]]]:
        """Split multiple images using batch prediction"""
        # Preprocess all images to grayscale
        processed_images = []
        for img in images:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            processed_images.append(img)
        
        # Get batch predictions
        predictions = list(self.predictor.predict_raw(processed_images))
        
        results = []
        for i, sample in enumerate(predictions):
            try:
                # Process each prediction similar to single image case
                outputs = getattr(sample, "outputs", None)
                if outputs is None:
                    raise ValueError("No outputs in prediction result")
                
                prediction_results = []
                if isinstance(outputs, tuple) and len(outputs) > 0:
                    prediction_results = outputs[0]
                elif isinstance(outputs, list):
                    prediction_results = outputs
                
                if not prediction_results:
                    raise ValueError("Empty prediction results")
                
                prediction = prediction_results[0] if isinstance(prediction_results, list) else prediction_results
                
                if not hasattr(prediction, 'prediction') or not prediction.prediction:
                    raise ValueError("No prediction data available")
                
                pred_obj = prediction.prediction
                char_images = []
                
                # Use hybrid approach for batch processing
                if hasattr(pred_obj, 'sentence') and pred_obj.sentence:
                    predicted_text = pred_obj.sentence
                    confidence = getattr(pred_obj, 'avg_char_probability', 0.5)
                    
                    # Use simple splitting with predicted text
                    temp_char_images = self._split_with_simple_method(processed_images[i], predicted_text)
                    char_images = [(char_img, char_text, confidence) for char_img, char_text in temp_char_images]
                
                if not char_images:
                    logger.warning(f"No valid characters found in image {i}")
                    # Create a dummy result to maintain batch structure
                    char_images = [(processed_images[i], '', 0.0)]
                
                results.append(char_images)
                
            except Exception as e:
                logger.warning(f"Failed to process image {i} in batch: {e}")
                # Fallback: create dummy result
                results.append([(processed_images[i], '', 0.0)])
        
        return results
    
    def _split_with_simple_method(self, image: np.ndarray, target_text: str) -> List[Tuple[np.ndarray, str, float]]:
        """Fallback simple splitting using equal-width division"""
        char_images = []
        num_chars = len(target_text)
        
        if num_chars == 0:
            return char_images
        
        width_per_char = image.shape[1] // num_chars
        
        for i, char in enumerate(target_text):
            # Calculate character boundaries
            char_start = i * width_per_char
            char_end = (i + 1) * width_per_char if i < num_chars - 1 else image.shape[1]
            
            # Extract character image
            char_img = image[:, char_start:char_end]
            
            # Apply width constraints
            char_img = self._apply_width_constraints(char_img)
            
            # Use default confidence for simple splitting
            confidence = 0.5
            char_images.append((char_img, char, confidence))
        
        return char_images
    
    def _apply_width_constraints(self, char_img: np.ndarray) -> np.ndarray:
        """Apply minimum and maximum width constraints to character image"""
        # Ensure minimum width
        if char_img.shape[1] < self.config.min_char_width:
            padding = (self.config.min_char_width - char_img.shape[1]) // 2
            char_img = np.pad(char_img, ((0, 0), (padding, self.config.min_char_width - char_img.shape[1] - padding)), 
                            mode='constant')
        
        # Limit maximum width (rare case, but handle it)
        elif char_img.shape[1] > self.config.max_char_width:
            # Center crop to maximum width
            start = (char_img.shape[1] - self.config.max_char_width) // 2
            char_img = char_img[:, start:start + self.config.max_char_width]
        
        return char_img
    
    def get_splitting_confidence(self, image: np.ndarray, target_text: str = None) -> float:
        """
        Get confidence score for the splitting operation
        
        Args:
            image: Input word image
            target_text: Optional target text for fallback
            
        Returns:
            Average confidence score for character splits
        """
        try:
            char_splits = self.split_word_to_chars(image, target_text)
            if not char_splits:
                return 0.0
            
            # Calculate average confidence
            confidences = [conf for _, _, conf in char_splits]
            return np.mean(confidences) if confidences else 0.0
            
        except Exception as e:
            logger.error(f"Failed to get splitting confidence: {e}")
            return 0.0
    
    def is_available(self) -> bool:
        """Check if the splitter is available and ready to use"""
        return self.predictor is not None or self.config.fallback_to_simple
    
    def get_name(self) -> str:
        """Get the name of this splitter"""
        return "Calamari Character Splitter"
    
    def clear_cache(self):
        """Clear the prediction cache"""
        if self._prediction_cache is not None:
            self._prediction_cache.clear()
            logger.info("Prediction cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self._prediction_cache is None:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self._prediction_cache),
            "memory_usage_mb": sum(len(v) for v in self._prediction_cache.values()) * 8 / (1024 * 1024)
        }
