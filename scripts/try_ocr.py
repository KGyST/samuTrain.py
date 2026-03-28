#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, 'src')

from bridge import OCRBridge

# Test OCR prediction
try:
    bridge = OCRBridge()
    print('OCR Bridge initialized')
    print(f'Model path: {bridge.model_path}')
    print(f'Model exists: {os.path.exists(bridge.model_path)}')
    
    # Test prediction on first image
    test_image = 'data/64_case/010001.bin.png'
    if os.path.exists(test_image):
        print(f'Testing prediction on: {test_image}')
        pred, conf = bridge.predict(test_image)
        print(f'Prediction: "{pred}"')
        print(f'Confidence: {conf}')
    else:
        print(f'Test image not found: {test_image}')
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
