#!/usr/bin/env python3
"""
Simple Calamari Test
Test if we can get basic calamari functionality working
"""

import sys
import os

# Add lib/calamari to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib', 'calamari'))

print("=== Simple Calamari Test ===")

try:
    import calamari_ocr
    print("✅ calamari_ocr imported successfully")
    print(f"   Version: {calamari_ocr.__version__}")
    
    # Try to import basic modules
    try:
        from calamari_ocr.ocr.predict.params import PredictionType
        print("✅ PredictionType imported")
    except Exception as e:
        print(f"⚠️  PredictionType import failed: {e}")
    
    # Try to create a simple predictor
    try:
        from calamari_ocr.ocr.predict.predictor import Predictor
        print("✅ Predictor imported")
        
        # Try to create predictor with empty model
        predictor = Predictor()
        print("✅ Predictor created successfully")
        
        # Try a simple prediction if we have an image
        data_dir = "data/64_case"
        if os.path.exists(data_dir):
            png_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
            if png_files:
                sample_image = os.path.join(data_dir, png_files[0])
                print(f"   Testing prediction on: {sample_image}")
                
                try:
                    result = predictor.predict([sample_image])
                    if result and len(result) > 0:
                        pred = result[0]
                        print(f"✅ Prediction successful!")
                        print(f"   Text: '{pred.sentence}'")
                        print(f"   Confidence: {pred.avg_char_probability:.3f}")
                    else:
                        print("⚠️  No prediction result")
                except Exception as e:
                    print(f"❌ Prediction failed: {e}")
            else:
                print("⚠️  No PNG files found")
        else:
            print("⚠️  data/64_case not found")
            
    except Exception as e:
        print(f"❌ Predictor creation failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ calamari_ocr import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
