"""
Test Character Splitter
------------------------
Comprehensive test script for the Calamari character splitter implementations.
Compares performance and accuracy between different splitting methods.
"""

import os
import sys
import time
import numpy as np
import cv2
from typing import List, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_test_image(image_path: str) -> np.ndarray:
    """Load and preprocess test image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img

def test_original_subprocess(image: np.ndarray, model_path: str, target_text: str) -> Tuple[List[Tuple[np.ndarray, str]], float]:
    """Test original subprocess method"""
    from scripts.try_calamari import _split_word_to_char_images_calamari
    
    start_time = time.time()
    char_images = _split_word_to_char_images_calamari(image, model_path)
    duration = time.time() - start_time
    
    return char_images, duration

def test_enhanced_splitter(image: np.ndarray, model_path: str, target_text: str) -> Tuple[List[Tuple[np.ndarray, str]], float]:
    """Test enhanced splitter"""
    from engines.calamari_splitter_v2 import CalamariCharacterSplitterV2, SplitterConfig
    
    config = SplitterConfig(
        use_native_api=True,
        use_subprocess_for_positions=True,
        fallback_to_simple=True,
        min_char_width=5,
        max_char_width=100,
        confidence_threshold=0.3,
        cache_predictions=True
    )
    
    splitter = CalamariCharacterSplitterV2(model_path, config)
    
    start_time = time.time()
    char_results = splitter.split_word_to_chars(image, target_text)
    duration = time.time() - start_time
    
    # Convert to expected format
    char_images = [(char_img, char_text) for char_img, char_text, _ in char_results]
    
    return char_images, duration

def test_native_only_splitter(image: np.ndarray, model_path: str, target_text: str) -> Tuple[List[Tuple[np.ndarray, str]], float]:
    """Test native API only splitter (no subprocess)"""
    from engines.calamari_splitter_v2 import CalamariCharacterSplitterV2, SplitterConfig
    
    config = SplitterConfig(
        use_native_api=True,
        use_subprocess_for_positions=False,  # Disable subprocess
        fallback_to_simple=True,
        min_char_width=5,
        max_char_width=100,
        confidence_threshold=0.3,
        cache_predictions=True
    )
    
    splitter = CalamariCharacterSplitterV2(model_path, config)
    
    start_time = time.time()
    char_results = splitter.split_word_to_chars(image, target_text)
    duration = time.time() - start_time
    
    # Convert to expected format
    char_images = [(char_img, char_text) for char_img, char_text, _ in char_results]
    
    return char_images, duration

def analyze_results(results: List[Tuple[str, List[Tuple[np.ndarray, str]], float]], target_text: str):
    """Analyze and compare results"""
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    print(f"\nTarget Text: '{target_text}'")
    print(f"Number of methods tested: {len(results)}")
    
    for method_name, char_images, duration in results:
        print(f"\n{method_name}:")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Characters extracted: {len(char_images)}")
        
        # Extract character texts
        extracted_texts = [char for _, char in char_images]
        print(f"  Extracted text: '{''.join(extracted_texts)}'")
        
        # Check accuracy
        if extracted_texts == list(target_text):
            print(f"  Accuracy: ✅ Perfect match")
        else:
            matches = sum(1 for a, b in zip(extracted_texts, target_text) if a == b)
            print(f"  Accuracy: {matches}/{len(target_text)} characters match")
        
        # Show character details
        for i, (char_img, char) in enumerate(char_images):
            print(f"    Char {i+1}: '{char}' | Shape: {char_img.shape}")

def run_performance_test():
    """Run comprehensive performance test"""
    print("Calamari Character Splitter Performance Test")
    print("="*50)
    
    # Test configuration
    MODEL_PATH = r"E:\Git\samuTrain.py\models\generic_ocr_model\best.ckpt.json"
    IMAGE_PATH = r"E:\Git\samuTrain.py\data\single_case\010071.bin.png"
    TARGET_TEXT = "1963"
    
    # Load test image
    print(f"\nLoading test image: {IMAGE_PATH}")
    image = load_test_image(IMAGE_PATH)
    print(f"Image shape: {image.shape}")
    
    # Test all methods
    results = []
    
    # Method 1: Original subprocess
    print("\n1. Testing original subprocess method...")
    try:
        char_images, duration = test_original_subprocess(image, MODEL_PATH, TARGET_TEXT)
        results.append(("Original Subprocess", char_images, duration))
        print(f"   ✅ Completed in {duration:.3f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 2: Enhanced splitter (subprocess + native)
    print("\n2. Testing enhanced splitter (subprocess + native)...")
    try:
        char_images, duration = test_enhanced_splitter(image, MODEL_PATH, TARGET_TEXT)
        results.append(("Enhanced Splitter", char_images, duration))
        print(f"   ✅ Completed in {duration:.3f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 3: Native only
    print("\n3. Testing native-only splitter...")
    try:
        char_images, duration = test_native_only_splitter(image, MODEL_PATH, TARGET_TEXT)
        results.append(("Native Only", char_images, duration))
        print(f"   ✅ Completed in {duration:.3f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Analyze results
    analyze_results(results, TARGET_TEXT)
    
    # Performance comparison
    if len(results) > 1:
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        
        fastest_method = min(results, key=lambda x: x[2])
        slowest_method = max(results, key=lambda x: x[2])
        
        print(f"\nFastest method: {fastest_method[0]} ({fastest_method[2]:.3f}s)")
        print(f"Slowest method: {slowest_method[0]} ({slowest_method[2]:.3f}s)")
        
        speedup = slowest_method[2] / fastest_method[2]
        print(f"Speedup factor: {speedup:.1f}x")
        
        # Show relative performance
        print(f"\nRelative performance (fastest = 1.0x):")
        for method_name, _, duration in results:
            relative = duration / fastest_method[2]
            print(f"  {method_name}: {relative:.1f}x")

def test_batch_processing():
    """Test batch processing capabilities"""
    print("\n" + "="*50)
    print("BATCH PROCESSING TEST")
    print("="*50)
    
    try:
        from engines.calamari_splitter_v2 import CalamariCharacterSplitterV2, SplitterConfig
        
        MODEL_PATH = r"E:\Git\samuTrain.py\models\generic_ocr_model\best.ckpt.json"
        IMAGE_PATH = r"E:\Git\samuTrain.py\data\single_case\010071.bin.png"
        
        # Load image and create duplicates for batch test
        image = load_test_image(IMAGE_PATH)
        batch_images = [image] * 3  # Test with 3 identical images
        target_texts = ["1963"] * 3
        
        config = SplitterConfig(batch_size=3)
        splitter = CalamariCharacterSplitterV2(MODEL_PATH, config)
        
        print(f"\nTesting batch processing with {len(batch_images)} images...")
        start_time = time.time()
        
        # Test individual processing
        individual_results = []
        for img, target in zip(batch_images, target_texts):
            result = splitter.split_word_to_chars(img, target)
            individual_results.append(result)
        
        individual_time = time.time() - start_time
        
        # Test batch processing
        start_time = time.time()
        batch_results = splitter.split_batch(batch_images, target_texts)
        batch_time = time.time() - start_time
        
        print(f"Individual processing: {individual_time:.3f}s")
        print(f"Batch processing: {batch_time:.3f}s")
        
        if batch_time > 0:
            speedup = individual_time / batch_time
            print(f"Batch speedup: {speedup:.1f}x")
        
        print(f"✅ Batch processing test completed")
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")

def main():
    """Main test function"""
    print("Calamari Character Splitter Test Suite")
    print("="*50)
    
    # Check if model exists
    MODEL_PATH = r"E:\Git\samuTrain.py\models\generic_ocr_model\best.ckpt.json"
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    # Run performance test
    run_performance_test()
    
    # Run batch processing test
    test_batch_processing()
    
    print("\n" + "="*50)
    print("TEST SUITE COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()
