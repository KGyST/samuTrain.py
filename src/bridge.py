import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
from calamari_ocr.ocr.predict.predictor import MultiPredictor

import requests

import json

import os

import time

from typing import List, Optional, Dict, Any

from pathlib import Path

# Try to import calamari with version handling
try:
  import sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
  # Trainer not used in current implementation
  Trainer = None
  try:
    from calamari_ocr import __version__
    print(f"calamari-ocr version: {__version__}")
  except ImportError:
    print("calamari-ocr version not available, continuing anyway")
    # Monkey-patch version if missing
    import calamari_ocr
    calamari_ocr.__version__ = '2.3.0'
    print("Applied monkey-patch for calamari-ocr version")
except ImportError as e:
  print(f"Failed to import calamari_ocr: {e}")
  print("Please ensure calamari-ocr is properly installed")
  Trainer = None

class SamuTrainCallback:
  """Callback to capture training results and send to samuTrain backend"""
  
  def __init__(self, backend_url: str = "http://localhost:8000", confidence_threshold: float = 0.8):
    self.backend_url = backend_url
    self.confidence_threshold = confidence_threshold
    self.session = requests.Session()
    self.session.headers.update({'Content-Type': 'application/json'})
  
  def __call__(self, epoch: int, step: int, data: Dict[str, Any]):
    """Called during training to process results"""
    try:
      # Extract OCR results from training data
      if 'predictions' in data:
        for pred in data['predictions']:
          self._process_prediction(pred, epoch, step)
    except Exception as e:
      print(f"Error in callback: {e}")
  
  def _process_prediction(self, prediction: Dict[str, Any], epoch: int, step: int):
    """Process individual prediction"""
    try:
      # Extract required fields
      img_path = prediction.get('image_path', '')
      ocr_text = prediction.get('text', '')
      confidence = prediction.get('confidence', 0.0)
      gt_text = prediction.get('ground_truth', '')
      
      if not img_path or not ocr_text:
        return
      
      # Determine if this is a failset case
      is_failset = confidence < self.confidence_threshold
      
      # Send to backend
      case_data = {
        'img_path': img_path,
        'ocr_text': ocr_text,
        'confidence': confidence,
        'gt_text': gt_text,
        'is_failset': is_failset
      }
      
      response = self.session.post(f"{self.backend_url}/api/cases", json=case_data)
      if response.status_code == 200:
        print(f"Sent case {img_path} (conf: {confidence:.3f}, failset: {is_failset})")
      else:
        print(f"Failed to send case: {response.status_code} {response.text}")
        
    except Exception as e:
      print(f"Error processing prediction: {e}")

class SamuTrainBridge:
  """Bridge between calamari trainer and samuTrain backend"""
  
  def __init__(self, backend_url: str = "http://localhost:8000", confidence_threshold: float = 0.8):
    self.backend_url = backend_url
    self.confidence_threshold = confidence_threshold
    self.callback = SamuTrainCallback(backend_url, confidence_threshold)
  
  def test_backend_connection(self) -> bool:
    """Test connection to samuTrain backend"""
    try:
      response = requests.get(f"{self.backend_url}/api/health", timeout=5)
      return response.status_code == 200
    except Exception as e:
      print(f"Backend connection test failed: {e}")
      return False
  
  def create_trainer_with_callback(self, trainer_config: Dict[str, Any]) -> Optional[object]:
    """Create calamari trainer with samuTrain callback"""
    if Trainer is None:
      print("calamari Trainer not available")
      return None
    
    try:
      # Add callback to trainer configuration
      if 'callbacks' not in trainer_config:
        trainer_config['callbacks'] = []
      
      trainer_config['callbacks'].append(self.callback)
      
      # Create trainer
      trainer = Trainer(**trainer_config)
      print("Trainer created with samuTrain callback")
      return trainer
      
    except Exception as e:
      print(f"Failed to create trainer: {e}")
      return None
  
  def run_training_with_monitoring(self, trainer_config: Dict[str, Any], 
                                 dataset_paths: List[str]) -> bool:
    """Run training with samuTrain monitoring"""
    if not self.test_backend_connection():
      print("Backend not available, cannot start monitoring")
      return False
    
    trainer = self.create_trainer_with_callback(trainer_config)
    if trainer is None:
      return False
    
    try:
      print("Starting training with samuTrain monitoring...")
      print(f"Confidence threshold: {self.confidence_threshold}")
      print(f"Backend URL: {self.backend_url}")
      
      # Run training (this will depend on calamari's API)
      # This is a placeholder - actual implementation depends on calamari version
      if hasattr(trainer, 'train'):
        trainer.train(dataset_paths)
      elif hasattr(trainer, 'fit'):
        trainer.fit(dataset_paths)
      else:
        print("Unknown trainer API, cannot proceed")
        return False
      
      print("Training completed with monitoring")
      return True
      
    except Exception as e:
      print(f"Training failed: {e}")
      return False

def create_sample_bridge():
  """Create a sample bridge for testing"""
  bridge = SamuTrainBridge()
  
  # Test connection
  if bridge.test_backend_connection():
    print("✓ Backend connection successful")
  else:
    print("✗ Backend connection failed")
  
  # Sample trainer config (adjust based on your calamari version)
  sample_config = {
    'epochs': 10,
    'batch_size': 16,
    'learning_rate': 0.001,
    # Add other calamari-specific parameters
  }
  
  trainer = bridge.create_trainer_with_callback(sample_config)
  if trainer:
    print("✓ Trainer created successfully")
  else:
    print("✗ Trainer creation failed")
  
  return bridge

if __name__ == "__main__":
  # Test the bridge
  print("Testing samuTrain Bridge...")
  bridge = create_sample_bridge()
