#!/usr/bin/env python3
"""
Debug Calamari Installation
Test if calamari is working correctly
"""

import sys
import os

# Add lib to path for calamari
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

print("=== Calamari Debug Test ===")
print(f"Python path: {sys.path[:3]}...")

try:
    print("\n1. Testing basic calamari import...")
    import calamari_ocr
    print("✅ calamari_ocr imported successfully")
    print(f"   Version: {getattr(calamari_ocr, '__version__', 'Unknown')}")
    
except ImportError as e:
    print(f"❌ Failed to import calamari_ocr: {e}")
    sys.exit(1)

try:
    print("\n2. Testing predictor import...")
    from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams
    print("✅ Predictor imported successfully")
    
except ImportError as e:
    print(f"❌ Failed to import Predictor: {e}")
    sys.exit(1)

try:
    print("\n3. Testing prediction params import...")
    from calamari_ocr.ocr.predict.params import PredictionType
    print("✅ PredictionType imported successfully")
    
except ImportError as e:
    print(f"❌ Failed to import PredictionType: {e}")
    sys.exit(1)

try:
    print("\n4. Testing model file access...")
    model_path = os.path.join('lib', 'calamari', 'calamari_ocr', 'test', 'models', 'version3', '0.ckpt.h5')
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
    else:
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error checking model file: {e}")
    sys.exit(1)

try:
    print("\n5. Testing predictor initialization...")
    predictor = Predictor(PredictorParams(
        checkpoint=model_path,
        prediction_type=PredictionType.BEST
    ))
    print("✅ Predictor initialized successfully")
    
except Exception as e:
    print(f"❌ Failed to initialize predictor: {e}")
    sys.exit(1)

try:
    print("\n6. Testing prediction on sample image...")
    # Find a sample image
    data_dir = "data/64_case"
    if os.path.exists(data_dir):
        png_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        if png_files:
            sample_image = os.path.join(data_dir, png_files[0])
            print(f"   Using sample image: {sample_image}")
            
            prediction_result = predictor.predict([sample_image])
            
            if prediction_result and len(prediction_result) > 0:
                pred = prediction_result[0]
                print(f"✅ Prediction successful!")
                print(f"   Text: '{pred.sentence}'")
                print(f"   Confidence: {pred.avg_char_probability:.3f}")
            else:
                print("❌ No prediction result returned")
        else:
            print("❌ No PNG files found in data/64_case")
    else:
        print("❌ data/64_case directory not found")
        
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== All Calamari Tests Passed! ===")
