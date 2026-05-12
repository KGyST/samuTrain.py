"""
Debug script to test the Calamari subprocess method specifically
"""

import os
import sys
import cv2
import tempfile

# Add paths for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
sys.path.insert(0, script_dir)

from try_calamari import _split_word_to_char_images_calamari

def main():
    MODEL_PATH = r"E:\Git\samuTrain.py\models\generic_ocr_model\best.ckpt.json"
    IMAGE_PATH = r"E:\Git\samuTrain.py\data\single_case\010071.bin.png"
    
    # Load image
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Failed to load image: {IMAGE_PATH}")
        return
    
    print(f"📸 Loaded image: {IMAGE_PATH}")
    print(f"   Shape: {img.shape}")
    print(f"   Model: {MODEL_PATH}")
    print("=" * 60)
    
    print("\n🔧 Testing Calamari_Subprocess method...")
    print("-" * 40)
    
    try:
        char_images = _split_word_to_char_images_calamari(img, MODEL_PATH)
        print(f"✅ SUCCESS: Got {len(char_images)} character images")
        for i, (char_img, char) in enumerate(char_images):
            print(f"  Char {i+1}: '{char}' | Shape: {char_img.shape}")
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    main()
