"""
Calamari Character Splitter V2
-------------------------------
Enhanced implementation that combines native API for prediction with subprocess for position data.
Optimized for performance while maintaining accuracy.
"""

import os
import sys
import numpy as np
import logging
import tempfile
import json
import subprocess
import shutil
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import cv2

# Calamari imports
from calamari_ocr.ocr.predict.predictor import MultiPredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SplitterConfig:
    """Configuration for the character splitter"""
    use_native_api: bool = True
    use_subprocess_for_positions: bool = True
    fallback_to_simple: bool = False  # No fallback
    min_char_width: int = 5
    max_char_width: int = 100
    confidence_threshold: float = 0.5
    batch_size: int = 8
    cache_predictions: bool = True

class CalamariCharacterSplitterV2:
    """Enhanced Calamari character splitter with optimized hybrid approach"""
    
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
        
        # Initialize predictor for fast predictions
        if self.config.use_native_api:
            try:
                self.predictor = MultiPredictor.from_paths(checkpoints=[self.model_path])
                logger.info(f"Calamari predictor initialized with model: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to initialize predictor: {e}")
                self.predictor = None
    
    def split_word_to_chars(self, image: np.ndarray, target_text: str = None) -> List[Tuple[np.ndarray, str, float]]:
        """
        Split a word image into individual character images
        
        Args:
            image: Input word image as numpy array (grayscale)
            target_text: Not used (kept for compatibility)
            
        Returns:
            List of tuples: (character_image, character_text, confidence)
        """
        # Check cache first
        if self._prediction_cache is not None:
            img_hash = hash(image.tobytes())
            if img_hash in self._prediction_cache:
                return self._prediction_cache[img_hash]
        
        # Ensure image is in correct format
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Try subprocess with position data (most accurate)
        if self.config.use_subprocess_for_positions:
            try:
                char_images = self._split_with_subprocess_positions(image)
                if char_images:
                    # Cache result if enabled
                    if self._prediction_cache is not None:
                        img_hash = hash(image.tobytes())
                        self._prediction_cache[img_hash] = char_images
                    return char_images
            except Exception as e:
                logger.warning(f"Subprocess position splitting failed: {e}")
        
        # Method 2: Try native API with predicted text
        if self.config.use_native_api and self.predictor is not None:
            try:
                char_images = self._split_with_native_prediction(image)
                if char_images:
                    # Cache result if enabled
                    if self._prediction_cache is not None:
                        img_hash = hash(image.tobytes())
                        self._prediction_cache[img_hash] = char_images
                    return char_images
            except Exception as e:
                logger.warning(f"Native prediction splitting failed: {e}")
        
        raise ValueError("All splitting methods failed")
    
    def _split_with_subprocess_positions(self, image: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        """Split using subprocess for accurate position data"""
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            cv2.imwrite(tmp_img.name, image)
            img_path = tmp_img.name
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Use Calamari's predict script with extended data
            # Use the same Python executable that's running this script
            python_exe = sys.executable
            cmd = [
                python_exe, "-m", "calamari_ocr.scripts.predict",
                "--checkpoint", self.model_path,
                "--data.files", img_path,
                "--extended_prediction_data",
                "--extended_prediction_data_format", "json",
                "--output_dir", temp_dir,
                "--verbose" if logger.isEnabledFor(logging.DEBUG) else "--quiet"
            ]
            
            # Run the prediction
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=os.path.dirname(self.model_path))
            
            if result.returncode != 0:
                logger.error(f"Calamari predict failed: {result.stderr}")
                return []
            
            # Read the prediction JSON
            json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
            if not json_files:
                logger.error("No prediction JSON file found")
                return []
            
            with open(os.path.join(temp_dir, json_files[0]), 'r') as f:
                prediction_data = json.loads(f.read())
            
            char_images = []
            
            # Extract character images using global_start and global_end positions
            if 'predictions' in prediction_data and prediction_data['predictions']:
                prediction = prediction_data['predictions'][0]
                if 'positions' in prediction:
                    for pos in prediction['positions']:
                        if 'chars' in pos and pos['chars']:
                            char = pos['chars'][0]['char']
                            confidence = pos['chars'][0].get('probability', 0.5)
                            char_start = max(0, pos['global_start'])
                            char_end = min(image.shape[1], pos['global_end'])
                            
                            if char_start < char_end:
                                char_img = image[:, char_start:char_end]
                                
                                # Apply width constraints
                                char_img = self._apply_width_constraints(char_img)
                                
                                char_images.append((char_img, char, confidence))
                            else:
                                logger.warning(f"Invalid position for character '{char}': {char_start}-{char_end}")
            
            return char_images
            
        finally:
            # Clean up temporary files
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _split_with_native_prediction(self, image: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        """Split using native API for fast prediction"""
        # Get prediction
        predictions = list(self.predictor.predict_raw([image]))
        
        if not predictions:
            return []
        
        sample = predictions[0]
        outputs = getattr(sample, "outputs", None)
        
        if outputs is None:
            return []
        
        # Extract prediction results
        prediction_results = []
        if isinstance(outputs, tuple) and len(outputs) > 0:
            prediction_results = outputs[0]
        elif isinstance(outputs, list):
            prediction_results = outputs
        
        if not prediction_results:
            return []
        
        # Get the first prediction
        prediction = prediction_results[0] if isinstance(prediction_results, list) else prediction_results
        
        if not hasattr(prediction, 'prediction') or not prediction.prediction:
            return []
        
        pred_obj = prediction.prediction
        
        # Extract predicted text and confidence
        if hasattr(pred_obj, 'sentence') and pred_obj.sentence:
            predicted_text = pred_obj.sentence
            confidence = getattr(pred_obj, 'avg_char_probability', 0.5)
            
            # Use simple splitting with predicted text
            char_images = self._split_with_simple_method(image, predicted_text)
            
            # Update confidence for each character
            char_images = [(char_img, char_text, confidence) for char_img, char_text in char_images]
            
            return char_images
        
        return []
    
    def _split_with_simple_method(self, image: np.ndarray, target_text: str) -> List[Tuple[np.ndarray, str]]:
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
            
            char_images.append((char_img, char))
        
        return char_images
    
    def _apply_width_constraints(self, char_img: np.ndarray) -> np.ndarray:
        """Apply minimum and maximum width constraints to character image"""
        # Ensure minimum width
        if char_img.shape[1] < self.config.min_char_width:
            padding = (self.config.min_char_width - char_img.shape[1]) // 2
            char_img = np.pad(char_img, ((0, 0), (padding, self.config.min_char_width - char_img.shape[1] - padding)), 
                            mode='constant', constant_values=255)
        
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
        return (self.predictor is not None or 
                self.config.use_subprocess_for_positions)
    
    def get_name(self) -> str:
        """Get the name of this splitter"""
        return "Calamari Character Splitter V2"
    
    def clear_cache(self):
        """Clear the prediction cache"""
        if self._prediction_cache is not None:
            self._prediction_cache.clear()
            logger.info("Prediction cache cleared")
    
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
            return self._split_batch_with_calamari_api(images, target_texts)
        except Exception as e:
            raise RuntimeError(f"Batch splitting failed: {e}")
    
    def _split_batch_with_calamari_api(self, images: List[np.ndarray], target_texts: List[str] = None) -> List[List[Tuple[np.ndarray, str, float]]]:
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
                elif hasattr(outputs, '__iter__'):
                    prediction_results = list(outputs)[0]
                else:
                    prediction_results = outputs
                
                # Get predicted text
                predicted_text = ""
                if hasattr(prediction_results, 'sentence'):
                    predicted_text = prediction_results.sentence
                elif hasattr(prediction_results, 'prediction'):
                    predicted_text = prediction_results.prediction
                
                confidence = 0.0
                if hasattr(prediction_results, 'avg_char_probability'):
                    confidence = prediction_results.avg_char_probability
                
                # Use subprocess for positions if enabled
                if self.config.use_subprocess_for_positions:
                    char_images = self._split_with_subprocess_positions(processed_images[i], predicted_text)
                else:
                    # Use simple splitting with predicted text
                    target_text = target_texts[i] if target_texts and i < len(target_texts) else predicted_text
                    temp_char_images = self._split_with_simple_method(processed_images[i], target_text)
                    char_images = [(char_img, char_text, confidence) for char_img, char_text in temp_char_images]
                
                if not char_images:
                    raise RuntimeError(f"No valid characters found in image {i}")
                
                results.append(char_images)
                
            except Exception as e:
                raise RuntimeError(f"Failed to process image {i} in batch: {e}")
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self._prediction_cache is None:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self._prediction_cache),
            "memory_usage_mb": sum(len(v) for v in self._prediction_cache.values()) * 8 / (1024 * 1024)
        }
