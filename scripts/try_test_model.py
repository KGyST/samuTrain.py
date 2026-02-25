#!/usr/bin/env python3
import os
import sys
import time

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'

def validate_arguments(model_dir, image_path):
  """Validate input arguments"""
  
  # Check model directory
  if not os.path.exists(model_dir):
    print(f"❌ Model directory not found: {model_dir}")
    return False
  
  # Check for model files
  model_path = os.path.join(model_dir, "best.ckpt")
  json_path = model_path + ".json"
  
  if not os.path.exists(json_path):
    print(f"❌ Model not found: {json_path}")
    return False
  
  print(f"✅ Model found: {json_path}")
  
  # Check image file
  if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    return False
  
  # Check image extension
  valid_extensions = ['.png', '.jpg', '.jpeg', '.bin.png']
  if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
    print(f"❌ Invalid image extension. Supported: {valid_extensions}")
    return False
  
  print(f"✅ Image found: {image_path}")
  
  # Check for ground truth file
  gt_file = image_path.replace('.bin.png', '.gt.txt')
  if os.path.exists(gt_file):
    print(f"✅ Ground truth found: {gt_file}")
  else:
    print(f"⚠️ No ground truth file: {gt_file}")
  
  return True

def test_model(model_dir, image_path):
  """Test a trained model on a specific image"""
  
  print(f"🧪 Testing model: {model_dir}")
  print(f"📷 Image: {image_path}")
  
  # Import here to avoid tensorflow import issues
  from calamari_ocr.ocr.predict.predictor import Predictor
  from calamari_ocr.ocr.predict.params import PredictorParams
  import numpy as np
  from PIL import Image
  
  try:
    # Check if model exists
    model_path = os.path.join(model_dir, "best.ckpt")
    json_path = model_path + ".json"
    
    if not os.path.exists(json_path):
      print(f"❌ Model not found: {json_path}")
      return False
    
    # Load ground truth if available
    gt_file = image_path.replace('.bin.png', '.gt.txt')
    gt_text = ""
    if os.path.exists(gt_file):
      gt_text = open(gt_file, 'r', encoding='utf-8').read().strip()
      print(f"📄 Ground truth: '{gt_text}'")
    
    # Load the model
    params = PredictorParams(silent=True)
    predictor = Predictor.from_checkpoint(params, checkpoint=model_path)
    
    print("✅ Model loaded successfully")
    
    # Load and test image
    img = np.array(Image.open(image_path).convert('L'))
    print(f"📊 Image info: shape={img.shape}, dtype={img.dtype}, range={img.min()}-{img.max()}")
    
    # Predict
    start_time = time.time()
    for sample in predictor.predict_raw([img]):
      pred_time = time.time() - start_time
      sentence = sample.outputs.sentence
      confidence = sample.outputs.avg_char_probability
      
      print(f"🔮 Prediction: '{sentence}'")
      print(f"📊 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
      print(f"⏱️ Time: {pred_time:.3f}s")
      
      # Check if prediction is meaningful
      if len(sentence.strip()) > 0:
        print("✅ Non-empty prediction!")
        
        # Compare with ground truth if available
        if gt_text:
          if sentence.strip() == gt_text.strip():
            print("🎯 Perfect match!")
          else:
            print(f"⚠️ Mismatch: predicted '{sentence}' vs gt '{gt_text}'")
      else:
        print("❌ Empty prediction")
        
      break
    
    return True
    
  except Exception as e:
    print(f"❌ Model testing failed: {e}")
    import traceback
    traceback.print_exc()
    return False

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python try_test_model.py <model_directory> <image_path>")
    print("Example: python try_test_model.py models/fixed_model data/1962/01000b.bin.png")
    sys.exit(1)
  
  model_dir = sys.argv[1]
  image_path = sys.argv[2]
  
  # Validate arguments
  if not validate_arguments(model_dir, image_path):
    sys.exit(1)
  
  success = test_model(model_dir, image_path)
  
  if not success:
    sys.exit(1)
