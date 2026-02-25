#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
from PIL import Image
from typing import List, Dict, Any

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'

try:
  from calamari_ocr.ocr.predict.predictor import Predictor
  from calamari_ocr.ocr.predict.params import PredictorParams
  LIB_MODE = True
except ImportError as e:
  print(f"❌ Import failed: {e}")
  LIB_MODE = False

class DirectPredictorTest:
  def __init__(self):
    self.test_images = [
      os.path.join(project_root, 'data', '1962', '01000b.bin.png'),  # "244."
      os.path.join(project_root, 'data', '1962', '01004b.bin.png'),  # "Gödöllő"
    ]
    self.model_paths = [
      os.path.join(project_root, 'models', 'new_model', 'best.ckpt'),
      os.path.join(project_root, 'models', 'new_model_1962', 'best.ckpt'),
    ]
    self.results = []

  def test_image_exists(self):
    """Verify test images exist"""
    for img_path in self.test_images:
      if not os.path.exists(img_path):
        print(f"❌ Test image not found: {img_path}")
        return False
      print(f"✅ Found test image: {os.path.basename(img_path)}")
    return True

  def test_model_exists(self):
    """Verify model files exist"""
    for model_path in self.model_paths:
      json_path = model_path + '.json'
      ckpt_path = model_path
      if os.path.exists(json_path):
        print(f"✅ Found model JSON: {json_path}")
      elif os.path.exists(ckpt_path):
        print(f"✅ Found model checkpoint: {ckpt_path}")
      else:
        print(f"❌ Model not found: {model_path}")
        return False
    return True

  def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
    """Load and preprocess image for prediction"""
    img = np.array(Image.open(image_path).convert('L'))
    print(f"📷 Image: {os.path.basename(image_path)}")
    print(f"   Shape: {img.shape}, dtype: {img.dtype}")
    print(f"   Range: {img.min()} - {img.max()}")
    return img

  def test_predictor_config(self, config_name: str, params: PredictorParams, model_path: str):
    """Test a specific predictor configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    try:
      # Create predictor
      checkpoint_base = model_path.replace('.json', '')
      predictor = Predictor.from_checkpoint(params, checkpoint=checkpoint_base)
      print(f"✅ Predictor created successfully")
      
      # Test on all images
      for img_path in self.test_images:
        if not os.path.exists(img_path):
          print(f"❌ Skip missing image: {os.path.basename(img_path)}")
          continue
          
        img = self.load_and_preprocess_image(img_path)
        
        # Single prediction
        start_time = time.time()
        for sample in predictor.predict_raw([img]):
          pred_time = time.time() - start_time
          sentence = sample.outputs.sentence
          confidence = sample.outputs.avg_char_probability
          
          print(f"🔮 Prediction: '{sentence}'")
          print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
          print(f"   Time: {pred_time:.3f}s")
          
          # Store result
          self.results.append({
            'config': config_name,
            'model': os.path.basename(model_path),
            'image': os.path.basename(img_path),
            'prediction': sentence,
            'confidence': confidence,
            'time': pred_time,
            'empty': len(sentence.strip()) == 0
          })
          break
          
    except Exception as e:
      print(f"❌ Test failed: {e}")
      import traceback
      traceback.print_exc()
      self.results.append({
        'config': config_name,
        'model': os.path.basename(model_path),
        'image': 'ERROR',
        'prediction': 'ERROR',
        'confidence': 0.0,
        'time': 0.0,
        'empty': True,
        'error': str(e)
      })

  def run_all_tests(self):
    """Run comprehensive test suite"""
    if not LIB_MODE:
      print("❌ Calamari library not available")
      return
      
    if not self.test_image_exists():
      return
      
    if not self.test_model_exists():
      return
    
    # Test configurations
    configs = [
      # Basic configurations
      ("Default", PredictorParams(silent=True)),
      ("Verbose", PredictorParams(silent=False)),
      
      # Device configurations
      ("CPU Only", PredictorParams(silent=True)),
    ]
    
    # Run tests for each model and configuration
    for model_path in self.model_paths:
      json_path = model_path + '.json'
      if not os.path.exists(json_path):
        continue
        
      for config_name, params in configs:
        self.test_predictor_config(config_name, params, model_path)

  def print_summary(self):
    """Print test results summary"""
    print(f"\n{'='*80}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Group results by configuration
    config_groups = {}
    for result in self.results:
      config = result['config']
      if config not in config_groups:
        config_groups[config] = []
      config_groups[config].append(result)
    
    for config, results in config_groups.items():
      print(f"\n📊 {config}:")
      empty_count = sum(1 for r in results if r.get('empty', False))
      total_count = len([r for r in results if r['image'] != 'ERROR'])
      avg_time = np.mean([r['time'] for r in results if r['time'] > 0])
      avg_conf = np.mean([r['confidence'] for r in results if r['confidence'] > 0])
      
      print(f"   Empty predictions: {empty_count}/{total_count} ({empty_count/total_count*100:.1f}%)")
      print(f"   Average time: {avg_time:.3f}s")
      print(f"   Average confidence: {avg_conf:.3f}")
      
      # Show individual predictions
      for result in results:
        if result['image'] != 'ERROR':
          status = "❌" if result.get('empty', False) else "✅"
          print(f"   {status} {result['image']}: '{result['prediction']}' ({result['confidence']:.3f})")
    
    # Find best configuration
    non_empty_results = [r for r in self.results if not r.get('empty', False) and r['image'] != 'ERROR']
    if non_empty_results:
      best_result = max(non_empty_results, key=lambda x: x['confidence'])
      print(f"\n🏆 Best configuration: {best_result['config']}")
      print(f"   Model: {best_result['model']}")
      print(f"   Prediction: '{best_result['prediction']}' (confidence: {best_result['confidence']:.3f})")
    else:
      print(f"\n❌ No successful predictions found!")

if __name__ == "__main__":
  print("🧪 Starting Direct Predictor Tests...")
  tester = DirectPredictorTest()
  tester.run_all_tests()
  tester.print_summary()
  print(f"\n✅ Testing complete!")
