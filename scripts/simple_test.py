#!/usr/bin/env python3
"""
Simple Calamari Test for Python 3.13
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

print("=== Simple Calamari Test ===")

try:
    import calamari_ocr
    print("✅ calamari_ocr imported successfully")
    print(f"   Version: {calamari_ocr.__version__}")
    
    # Try to import predictor directly
    try:
        from calamari_ocr.ocr.predictor import Predictor
        print("✅ Predictor imported successfully")
        
        # Try to create a predictor
        predictor = Predictor()
        print("✅ Predictor created successfully")
        print("🎉 Calamari is working!")
        
    except Exception as e:
        print(f"❌ Predictor test failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ calamari_ocr import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
