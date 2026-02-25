#!/usr/bin/env python3
import os
import sys
import numpy as np
from PIL import Image

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'

from calamari_ocr.ocr.predict.predictor import Predictor
from calamari_ocr.ocr.predict.params import PredictorParams

def diagnose_model():
  """Diagnose why model returns empty predictions"""
  
  # Use the working model
  model_path = os.path.join(project_root, 'models', 'new_model', 'best.ckpt')
  test_image = os.path.join(project_root, 'data', '1962', '01000b.bin.png')
  
  print(f"🔍 Diagnosing model: {model_path}")
  print(f"📷 Test image: {test_image}")
  print(f"📄 Ground truth: {open(test_image.replace('.bin.png', '.gt.txt')).read().strip()}")
  
  # Load image
  img = np.array(Image.open(test_image).convert('L'))
  print(f"📊 Image info: shape={img.shape}, dtype={img.dtype}, range={img.min()}-{img.max()}")
  
  # Create predictor
  params = PredictorParams(silent=True)  # Keep silent to avoid verbose output issues
  predictor = Predictor.from_checkpoint(params, checkpoint=model_path)
  
  print("\n🔬 Deep diagnosis of prediction output:")
  
  # Get raw prediction
  for sample in predictor.predict_raw([img]):
    print(f"📦 Sample type: {type(sample)}")
    print(f"📦 Sample attributes: {[attr for attr in dir(sample) if not attr.startswith('_')]}")
    
    outputs = sample.outputs
    print(f"📤 Outputs type: {type(outputs)}")
    print(f"📤 Outputs attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
    
    # Check all prediction fields
    print(f"\n🔍 Prediction details:")
    print(f"  Sentence: '{outputs.sentence}'")
    print(f"  Labels: {outputs.labels}")
    print(f"  Labels length: {len(outputs.labels)}")
    print(f"  Total probability: {outputs.total_probability}")
    print(f"  Avg char probability: {outputs.avg_char_probability}")
    print(f"  Positions count: {len(outputs.positions)}")
    
    # Check positions (this is where character info lives)
    if outputs.positions:
      print(f"\n📍 Position analysis:")
      for i, pos in enumerate(outputs.positions):
        print(f"  Position {i}:")
        print(f"    Local range: {pos.local_start}-{pos.local_end}")
        print(f"    Global range: {pos.global_start}-{pos.global_end}")
        print(f"    Chars count: {len(pos.chars)}")
        
        if pos.chars:
          print(f"    Characters:")
          for j, char in enumerate(pos.chars):
            print(f"      {j}: '{char.char}' (label={char.label}, prob={char.probability})")
        else:
          print(f"    ❌ No characters found!")
    else:
      print(f"❌ No positions found!")
    
    # Check logits (raw network output)
    if hasattr(outputs, 'logits') and outputs.logits is not None:
      logits = outputs.logits
      print(f"\n🧮 Logits analysis:")
      print(f"  Shape: {logits.shape}")
      print(f"  Min/Max: {logits.min():.3f} / {logits.max():.3f}")
      print(f"  Mean: {logits.mean():.3f}")
      print(f"  Std: {logits.std():.3f}")
      
      # Check if logits are all zeros or very small
      if np.allclose(logits, 0):
        print(f"  ❌ All logits are zero!")
      elif logits.max() < 0.1:
        print(f"  ⚠️ All logits are very small (< 0.1)")
    else:
      print(f"❌ No logits found!")
    
    break

if __name__ == "__main__":
  print("🩺 Starting empty prediction diagnosis...")
  diagnose_model()
  print(f"\n✅ Diagnosis complete!")
