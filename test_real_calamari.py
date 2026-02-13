#!/usr/bin/env python3
"""
Test Real Calamari Prediction
"""

import sys
sys.path.insert(0, 'src')

from engines.calamari_learner import CalamariLearner
import os

def main():
    print("=== Testing Real Calamari Prediction ===")
    
    learner = CalamariLearner()
    print(f"Learner available: {learner.is_available()}")
    print(f"Learner name: {learner.get_name()}")
    
    if not learner.is_available():
        print("❌ Calamari not available")
        return
    
    test_image = 'data/single_case/test001.bin.png'
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"Testing prediction on: {test_image}")
    
    try:
        text, confidence = learner.predict(test_image)
        print("✅ Real Calamari Prediction:")
        print(f"   Text: '{text}'")
        print(f"   Confidence: {confidence:.3f}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
