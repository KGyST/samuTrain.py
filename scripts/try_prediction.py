#!/usr/bin/env python3
import os
import sys

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from bridge import OCRBridge
import numpy as np
from PIL import Image

def test_prediction():
  print("Testing OCR prediction...")
  
  # Initialize bridge
  ocr_bridge = OCRBridge()
  print(f"Bridge info: {ocr_bridge.get_learner_info()}")
  
  # Test on one of the 1962 images
  test_image = os.path.join(project_root, 'data', '1962', '_0001_01006f.bin.png')
  print(f"Testing on: {test_image}")
  
  if not os.path.exists(test_image):
    print(f"❌ Test image not found: {test_image}")
    return
  
  try:
    pred, conf = ocr_bridge.predict(test_image)
    print(f"Prediction: '{pred}'")
    print(f"Confidence: {conf}")
    print(f"Confidence %: {conf * 100:.2f}%")
    
    # Also test the raw predictor
    if hasattr(ocr_bridge.learner, 'predictor') and ocr_bridge.learner.predictor:
      print("\nTesting raw predictor...")
      img = np.array(Image.open(test_image).convert('L'))
      print(f"Image shape: {img.shape}")
      print(f"Image dtype: {img.dtype}")
      print(f"Image min/max: {img.min()}/{img.max()}")
      
      for sample in ocr_bridge.learner.predictor.predict_raw([img]):
        print(f"Raw Sample object: {sample}")
        print(f"Sample attributes: {dir(sample)}")
        print(f"Sample type: {type(sample)}")
        print(f"Sample outputs: {sample.outputs}")
        print(f"Sample outputs type: {type(sample.outputs)}")
        if sample.outputs:
          print(f"Sample outputs type: {type(sample.outputs)}")
          print(f"Sample outputs[0]: {sample.outputs[0]}")
        print(f"Sample targets: {sample.targets}")
        break
        
  except Exception as e:
    print(f"❌ Prediction test failed: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
  test_prediction()
