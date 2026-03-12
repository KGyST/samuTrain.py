#!/usr/bin/env python3
"""
Test OCR prediction
"""
import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

from src.bridge import OCRBridge

# Test OCR prediction
ocr = OCRBridge()
print(f'Learner info: {ocr.get_learner_info()}')

# Test with a sample image
test_image = 'data/64_case/010001.bin.png'
if os.path.exists(test_image):
    pred, conf = ocr.predict(test_image)
    print(f'Test prediction: "{pred}" (confidence: {conf:.3f})')
else:
    print(f'Test image not found: {test_image}')
